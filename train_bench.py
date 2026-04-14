import torch
import time
import sys
import os
import random
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Callable


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama

DEVICE = "cuda"
CACHE_DIR = "/root/train/grokadamw/cached_tokens"


@dataclass
class TrainConfig:
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 512
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    eval_every: int = 100
    gradient_clipping: float = 1.0  # 0 = disabled
    n_sequences: int = 20000
    eval_ratio: float = 0.1  # held-out fraction
    eval_batches: int = 10
    eval_batch_size: int = 16
    seed: int = 42


@dataclass
class OptimizerSpec:
    """One optimizer to benchmark."""

    name: str
    factory: Callable  # (params, cfg) -> optimizer
    description: str = ""


def create_model(cfg: TrainConfig):
    config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_model * 4,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_head,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16, device=DEVICE)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def preprocess_and_cache(
    tokenizer, cfg: TrainConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize and split into train/eval. Returns (train_x, train_y, eval_x, eval_y)."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_x_path = os.path.join(CACHE_DIR, "train_x.pt")
    train_y_path = os.path.join(CACHE_DIR, "train_y.pt")
    eval_x_path = os.path.join(CACHE_DIR, "eval_x.pt")
    eval_y_path = os.path.join(CACHE_DIR, "eval_y.pt")

    if all(
        os.path.exists(p)
        for p in [train_x_path, train_y_path, eval_x_path, eval_y_path]
    ):
        print(f"Loading cached tokens from {CACHE_DIR}...")
        train_x = torch.load(train_x_path, weights_only=True)
        train_y = torch.load(train_y_path, weights_only=True)
        eval_x = torch.load(eval_x_path, weights_only=True)
        eval_y = torch.load(eval_y_path, weights_only=True)
        print(f"  train: {train_x.shape[0]}, eval: {eval_x.shape[0]} sequences")
        return train_x, train_y, eval_x, eval_y

    print(f"Tokenizing dataset and caching to {CACHE_DIR}...")
    ds = load_dataset("C10X/finepdf2", split="train", streaming=True)

    xs, ys = [], []
    buf = []
    total = 0

    for sample in ds:
        text = sample["text"]
        if not text or len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(tokens)

        while len(buf) >= cfg.max_seq_len + 1:
            chunk = buf[: cfg.max_seq_len + 1]
            buf = buf[cfg.max_seq_len :]
            xs.append(torch.tensor(chunk[:-1], dtype=torch.long))
            ys.append(torch.tensor(chunk[1:], dtype=torch.long))
            total += 1
            if total >= cfg.n_sequences:
                break

        if total >= cfg.n_sequences:
            break

        if total % 2000 == 0 and total > 0:
            print(f"  tokenized {total}/{cfg.n_sequences} sequences...")

    all_x = torch.stack(xs)
    all_y = torch.stack(ys)

    # deterministic split
    n_total = all_x.shape[0]
    n_eval = max(1, int(n_total * cfg.eval_ratio))
    n_train = n_total - n_eval

    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=rng)

    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    train_x, train_y = all_x[train_idx], all_y[train_idx]
    eval_x, eval_y = all_x[eval_idx], all_y[eval_idx]

    torch.save(train_x, train_x_path)
    torch.save(train_y, train_y_path)
    torch.save(eval_x, eval_x_path)
    torch.save(eval_y, eval_y_path)
    print(f"  cached train: {n_train}, eval: {n_eval} sequences")
    return train_x, train_y, eval_x, eval_y


@torch.no_grad()
def evaluate(model, eval_x, eval_y, n_batches=10, batch_size=16):
    model.eval()
    losses = []
    n_available = eval_x.shape[0]
    actual_batches = min(n_batches, n_available // batch_size)
    if actual_batches == 0:
        actual_batches = 1
        batch_size = n_available

    indices = torch.randperm(n_available)[: actual_batches * batch_size]
    for i in range(0, len(indices), batch_size):
        idx = indices[i : i + batch_size]
        x = eval_x[idx].to(DEVICE)
        y = eval_y[idx].to(DEVICE)
        output = model(x, labels=y)
        losses.append(output.loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def optimizer_memory(optimizer):
    gpu_bytes = 0
    cpu_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                b = v.element_size() * v.nelement()
                if v.is_cuda:
                    gpu_bytes += b
                else:
                    cpu_bytes += b
    return gpu_bytes / 1024**2, cpu_bytes / 1024**2


def estimate_tflops(n_params, seq_len, batch_size, ms_per_step):
    if ms_per_step <= 0:
        return 0.0
    fwd_flops = 2 * n_params * seq_len * batch_size
    bwd_flops = 4 * n_params * seq_len * batch_size
    total_flops = fwd_flops + bwd_flops
    return total_flops / (ms_per_step / 1000) / 1e12


def train(
    name: str,
    model,
    optimizer,
    train_x,
    train_y,
    eval_x,
    eval_y,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """Train and return summary dict for comparison."""
    model.train()

    n_params, n_trainable = count_params(model)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_tflops_peak = 94.81  # RTX 5060 Ti bf16

    print(f"\n{'=' * 145}")
    print(f"  {name}")
    print(
        f"  params: {n_params / 1e6:.2f}M | Llama ({cfg.d_model}d, {cfg.n_layer}L, {cfg.n_head}h)"
        f" | seq={cfg.max_seq_len}, bs={cfg.batch_size}, train={train_x.shape[0]}, eval={eval_x.shape[0]} seqs"
    )
    print(f"{'=' * 145}")
    header = (
        f"{'step':>5} {'loss':>8} {'eval':>8} {'gap':>7}"
        f" {'data':>6} {'fwd':>6} {'bwd':>6} {'opt':>6} {'tot':>6}"
        f" {'tok/s':>8} {'TFLOPS':>7} {'MFU%':>5} {'VRAM':>7}"
        f" {'opt(G/C)':>12} {'grad_n':>8} {'p_norm':>7}"
    )
    print(header)
    print("-" * 145)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    n_train = train_x.shape[0]
    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_train, generator=rng)
    data_idx = 0

    fwd_acc = bwd_acc = opt_acc = data_acc = 0.0
    tokens_acc = 0
    step_t0 = time.perf_counter()

    history = []  # (step, train_loss, eval_loss)

    for step in range(1, cfg.max_steps + 1):
        # --- data loading ---
        td = time.perf_counter()
        if data_idx + cfg.batch_size > n_train:
            perm = torch.randperm(n_train, generator=rng)
            data_idx = 0
        idx = perm[data_idx : data_idx + cfg.batch_size]
        data_idx += cfg.batch_size
        batch_x = train_x[idx].to(DEVICE, non_blocking=True)
        batch_y = train_y[idx].to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        data_acc += (time.perf_counter() - td) * 1000

        B, T = batch_x.shape
        tokens_acc += B * T

        # --- forward ---
        torch.cuda.synchronize()
        tf = time.perf_counter()
        output = model(batch_x, labels=batch_y)
        loss = output.loss
        torch.cuda.synchronize()
        fwd_acc += (time.perf_counter() - tf) * 1000

        # --- backward ---
        torch.cuda.synchronize()
        tb = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_acc += (time.perf_counter() - tb) * 1000

        # --- gradient clipping ---
        if cfg.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)

        # --- grad/param norms (before step, while grads exist) ---
        if step % cfg.eval_every == 0:
            total_grad_norm = 0.0
            total_param_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.float().norm().item() ** 2
                total_param_norm += p.data.float().norm().item() ** 2
            grad_norm = total_grad_norm**0.5
            param_norm = total_param_norm**0.5

        # --- optimizer step ---
        torch.cuda.synchronize()
        to = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_acc += (time.perf_counter() - to) * 1000

        optimizer.zero_grad(set_to_none=True)

        # --- logging ---
        if step % cfg.eval_every == 0:
            # VRAM: read peak from training steps, before eval pollutes it
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            opt_gpu, opt_cpu = optimizer_memory(optimizer)

            # timing
            elapsed_s = time.perf_counter() - step_t0
            total_ms = elapsed_s / cfg.eval_every * 1000
            fwd_avg = fwd_acc / cfg.eval_every
            bwd_avg = bwd_acc / cfg.eval_every
            opt_avg = opt_acc / cfg.eval_every
            data_avg = data_acc / cfg.eval_every
            tok_per_s = tokens_acc / elapsed_s if elapsed_s > 0 else 0

            compute_ms = fwd_avg + bwd_avg
            tflops = estimate_tflops(
                n_params, cfg.max_seq_len, cfg.batch_size, compute_ms
            )
            mfu = (tflops / gpu_tflops_peak) * 100

            # eval on held-out data
            eval_loss = evaluate(
                model,
                eval_x,
                eval_y,
                n_batches=cfg.eval_batches,
                batch_size=cfg.eval_batch_size,
            )
            gap = eval_loss - loss.item()
            history.append((step, loss.item(), eval_loss))

            print(
                f"{step:>5} {loss.item():>8.4f} {eval_loss:>8.4f} {gap:>7.4f}"
                f" {data_avg:>6.1f} {fwd_avg:>6.1f} {bwd_avg:>6.1f} {opt_avg:>6.1f} {total_ms:>6.1f}"
                f" {tok_per_s:>8.0f} {tflops:>7.1f} {mfu:>5.1f} {vram_mb:>6.0f}MB"
                f" {opt_gpu:>5.0f}/{opt_cpu:>5.0f}MB {grad_norm:>8.2f} {param_norm:>7.2f}"
            )

            # reset for next window — after eval so eval time doesn't leak
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            step_t0 = time.perf_counter()
            fwd_acc = bwd_acc = opt_acc = data_acc = 0.0
            tokens_acc = 0

    return {
        "name": name,
        "history": history,
        "final_train_loss": history[-1][1] if history else float("inf"),
        "final_eval_loss": history[-1][2] if history else float("inf"),
    }


def print_comparison(results: List[Dict[str, Any]]):
    """Print side-by-side summary of all optimizer runs."""
    print(f"\n{'=' * 90}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Optimizer':<25} {'Final Train':>12} {'Final Eval':>12} {'Gap':>10} {'Best Eval':>12}"
    )
    print("-" * 90)

    for r in results:
        best_eval = min(h[2] for h in r["history"]) if r["history"] else float("inf")
        gap = r["final_eval_loss"] - r["final_train_loss"]
        print(
            f"{r['name']:<25} {r['final_train_loss']:>12.4f} {r['final_eval_loss']:>12.4f}"
            f" {gap:>10.4f} {best_eval:>12.4f}"
        )

    print(f"{'=' * 90}")

    # step-by-step comparison table
    if len(results) > 1 and all(r["history"] for r in results):
        steps = [h[0] for h in results[0]["history"]]
        print(f"\n  Eval Loss per Step")
        print(f"{'step':>6}", end="")
        for r in results:
            print(f"  {r['name']:>18}", end="")
        print()
        print("-" * (6 + 20 * len(results)))

        for i, step in enumerate(steps):
            print(f"{step:>6}", end="")
            for r in results:
                if i < len(r["history"]):
                    print(f"  {r['history'][i][2]:>18.4f}", end="")
                else:
                    print(f"  {'---':>18}", end="")
            print()


# ─────────────────────────────────────────────
#  Optimizer registry — add new optimizers here
# ─────────────────────────────────────────────


def get_optimizer_specs() -> List[OptimizerSpec]:
    """Define all optimizers to benchmark. Edit this function to add/remove."""

    specs = []

    # --- GrokAdamW (local fork) ---
    def make_grokadamw(params, cfg):
        for mod in list(sys.modules.keys()):
            if "grokadamw" in mod.lower():
                del sys.modules[mod]
        sys.path.insert(0, os.path.dirname(__file__))
        from grokadamw import GrokAdamW
        import grokadamw as _ref

        print(f"  [GrokAdamW from: {_ref.__file__}]")
        return GrokAdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    specs.append(
        OptimizerSpec(
            name="GrokAdamW",
            factory=make_grokadamw,
            description="Local GrokAdamW fork with foreach ops",
        )
    )

    # --- Baseline AdamW (torch) ---
    def make_adamw(params, cfg):
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,
        )

    specs.append(
        OptimizerSpec(
            name="AdamW (fused)",
            factory=make_adamw,
            description="PyTorch native fused AdamW",
        )
    )

    # --- Uncomment to add more ---
    # def make_custom(params, cfg):
    #     from some_module import SomeOptimizer
    #     return SomeOptimizer(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # specs.append(OptimizerSpec(name="Custom", factory=make_custom))

    return specs


def main():
    cfg = TrainConfig()

    print(
        f"torch={torch.__version__}, transformers={__import__('transformers').__version__}"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = tokenizer.vocab_size

    # Liger monkey-patch BEFORE any model instantiation
    # fused_linear_cross_entropy=False: incompatible with PyTorch 2.9.1 backward
    apply_liger_kernel_to_llama()

    train_x, train_y, eval_x, eval_y = preprocess_and_cache(tokenizer, cfg)

    optimizer_specs = get_optimizer_specs()
    results = []

    for spec in optimizer_specs:
        print(f"\n>>> Preparing: {spec.name}")
        if spec.description:
            print(f"    {spec.description}")

        seed_everything(cfg.seed)
        model = create_model(cfg)
        optimizer = spec.factory(model.parameters(), cfg)

        result = train(
            spec.name,
            model,
            optimizer,
            train_x,
            train_y,
            eval_x,
            eval_y,
            cfg,
        )
        results.append(result)

        del model, optimizer
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    print_comparison(results)

    print(f"\n{'=' * 90}")
    print("DONE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
