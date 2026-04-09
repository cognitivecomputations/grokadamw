import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from grokadamw.grokadamw import GrokAdamW

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


def create_large(hidden=1024, layers=8):
    modules = []
    for i in range(layers):
        if i > 0:
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden, hidden))
    return nn.Sequential(*modules)


def create_transformer(hidden=256, heads=4, layers=4, vocab_size=1000):
    class TransformerModel(nn.Module):
        def __init__(self):
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

    return TransformerModel()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_step(model, optimizer, x_fn, n_steps=300, warmup=30):
    def step():
        x = x_fn()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for _ in range(warmup):
        step()

    times = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return np.median(times), np.mean(times), mem


def run():
    if not torch.cuda.is_available():
        print("CUDA required")
        return

    print(f"torch={torch.__version__}, triton={'yes' if HAS_TRITON else 'no'}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from grokadamw._triton_fused import _TRITON_AVAILABLE

    print(f"Triton fused kernel available: {_TRITON_AVAILABLE}")
    print()

    models = [
        (
            "Small MLP (6 tensors)",
            create_mlp,
            lambda: torch.randn(32, 512, device="cuda"),
        ),
        (
            "Large MLP (16 tensors)",
            create_large,
            lambda: torch.randn(32, 1024, device="cuda"),
        ),
        (
            "Transformer (51 tensors)",
            create_transformer,
            lambda: torch.randint(0, 1000, (32, 64), device="cuda"),
        ),
    ]

    configs = [
        ("PyTorch GPU states", dict(cpu_offload=False), False),
        ("Triton fused kernel", dict(cpu_offload=False), True),
    ]

    N_RUNS = 3

    for model_name, model_fn, x_fn in models:
        m = model_fn().cuda()
        n_params = count_params(m)
        n_tensors = len(list(m.parameters()))
        del m
        torch.cuda.empty_cache()
        print(
            f"=== {model_name} ({n_params / 1e6:.2f}M params, {n_tensors} tensors) ==="
        )

        for config_name, config, use_triton in configs:
            if use_triton and not _TRITON_AVAILABLE:
                continue

            import grokadamw.grokadamw as optmod

            original = optmod._TRITON_AVAILABLE
            optmod._TRITON_AVAILABLE = use_triton

            medians = []
            means = []
            mems = []
            for run_idx in range(N_RUNS):
                torch.manual_seed(0)
                model = model_fn().cuda()
                torch.cuda.reset_peak_memory_stats()
                optimizer = GrokAdamW(model.parameters(), **config)
                med, avg, mem = benchmark_step(
                    model, optimizer, x_fn, n_steps=300, warmup=30
                )
                medians.append(med)
                means.append(avg)
                mems.append(mem)
                del model, optimizer
                torch.cuda.empty_cache()

            optmod._TRITON_AVAILABLE = original

            final_med = np.median(medians)
            final_mean = np.mean(means)
            final_mem = max(mems)
            print(
                f"  {config_name:30s}  median={final_med:.3f} ms/step  mean={final_mean:.3f} ms/step  {final_mem:.1f} MB peak"
            )

        print()


if __name__ == "__main__":
    run()
