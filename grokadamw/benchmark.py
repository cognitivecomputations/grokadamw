import torch
import torch.nn as nn
import time

from grokadamw import GrokAdamW

try:
    import triton
    from triton.testing import do_bench

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def create_mlp(hidden=512, layers=3):
    modules = []
    for i in range(layers):
        if i > 0:
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden, hidden))
    return nn.Sequential(*modules)


class TransformerModel(nn.Module):
    def __init__(self, hidden, heads, layers, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 4,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


def create_transformer(hidden=256, heads=4, layers=4, vocab_size=1000):
    return TransformerModel(hidden, heads, layers, vocab_size)


def create_large(hidden=1024, layers=8):
    modules = []
    for i in range(layers):
        if i > 0:
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden, hidden))
    return nn.Sequential(*modules)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_step(model, optimizer, x_fn, n_steps=100, warmup=10):
    def step():
        x = x_fn()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for _ in range(warmup):
        step()

    torch.cuda.reset_peak_memory_stats()

    if HAS_TRITON:
        ms = do_bench(step, warmup=0, rep=n_steps)
    else:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / n_steps * 1000

    mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return ms, mem


def numerical_equivalence_test(
    model_fn, x_fn, n_steps=50, atol=0.01, rtol=0.05, dtype=torch.float32
):
    torch.manual_seed(42)
    model_ref = model_fn().cuda().to(dtype)
    state = {k: v.clone() for k, v in model_ref.state_dict().items()}

    torch.manual_seed(42)
    model_gpu = model_fn().cuda().to(dtype)
    model_gpu.load_state_dict(state)

    opt_ref = GrokAdamW(model_ref.parameters(), cpu_offload=True)
    opt_gpu = GrokAdamW(model_gpu.parameters(), cpu_offload=False)

    for step in range(n_steps):
        torch.manual_seed(step + 1000)
        x_ref = x_fn().to(dtype)
        x_gpu = x_ref.clone()

        loss_ref = model_ref(x_ref).sum()
        loss_ref.backward()
        opt_ref.step()
        model_ref.zero_grad()

        loss_gpu = model_gpu(x_gpu).sum()
        loss_gpu.backward()
        opt_gpu.step()
        model_gpu.zero_grad()

        for j, (p_ref, p_gpu) in enumerate(
            zip(model_ref.parameters(), model_gpu.parameters())
        ):
            if not torch.allclose(p_ref.float(), p_gpu.float(), atol=atol, rtol=rtol):
                print(
                    f"  FAIL: step {step}, param {j}, max_diff={((p_ref - p_gpu).abs().max()).item():.2e}"
                )
                return False

    return True


def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA required for benchmarks")
        return

    models = [
        ("MLP ~790K", create_mlp, lambda: torch.randn(32, 512, device="cuda")),
        (
            "Transformer ~2M",
            create_transformer,
            lambda: torch.randint(0, 1000, (32, 64), device="cuda"),
        ),
        ("Large ~10M", create_large, lambda: torch.randn(32, 1024, device="cuda")),
    ]

    configs = [
        ("CPU states (cpu_offload=True)", dict(cpu_offload=True)),
        ("GPU states (cpu_offload=False)", dict(cpu_offload=False)),
    ]

    if not HAS_TRITON:
        print("[INFO] Triton not available, GPU states will use PyTorch fallback\n")

    for model_name, model_fn, x_fn in models:
        n_params = count_params(model_fn().cuda())
        print(f"=== {model_name} ({n_params / 1e6:.2f}M params) ===")

        for config_name, config in configs:
            torch.manual_seed(0)
            model = model_fn().cuda()
            torch.cuda.reset_peak_memory_stats()

            optimizer = GrokAdamW(model.parameters(), **config)

            ms, mem = benchmark_step(model, optimizer, x_fn, n_steps=100, warmup=10)
            print(f"  {config_name:35s}  {ms:.3f} ms/step  {mem:.1f} MB peak")

        print()

    print("=== Numerical Equivalence ===")
    for dtype, dtype_name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        result = numerical_equivalence_test(
            create_mlp,
            lambda: torch.randn(32, 512, device="cuda"),
            n_steps=50,
            dtype=dtype,
        )
        status = "PASS" if result else "FAIL"
        print(f"  CPU vs GPU states ({dtype_name}): {status}")


if __name__ == "__main__":
    run_benchmarks()
