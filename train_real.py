import torch
import time
import sys
import os
import random
import numpy as np
import importlib
import importlib.util
import math
import logging
from dataclasses import dataclass


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


seed_everything(42)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

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
    gradient_clipping: float = 0.0
    n_sequences: int = 20000


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


def preprocess_and_cache(tokenizer, cfg):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_x_path = os.path.join(CACHE_DIR, "all_x.pt")
    all_y_path = os.path.join(CACHE_DIR, "all_y.pt")

    if os.path.exists(all_x_path) and os.path.exists(all_y_path):
        print(f"Loading cached tokens from {CACHE_DIR}...")
        all_x = torch.load(all_x_path, weights_only=True)
        all_y = torch.load(all_y_path, weights_only=True)
        print(f"  loaded {all_x.shape[0]} sequences")
        return all_x, all_y

    print(f"Tokenizing dataset and caching to {CACHE_DIR}...")
    ds = load_dataset("C10X/finepdf2", split="train", streaming=True)

    xs = []
    ys = []
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

        if total % 1000 == 0 and total > 0:
            print(f"  tokenized {total}/{cfg.n_sequences} sequences...")

    all_x = torch.stack(xs)
    all_y = torch.stack(ys)
    torch.save(all_x, all_x_path)
    torch.save(all_y, all_y_path)
    print(f"  cached {all_x.shape[0]} sequences to {CACHE_DIR}")
    return all_x, all_y


@torch.no_grad()
def evaluate(model, all_x, all_y, n_batches=10, batch_size=16):
    losses = []
    indices = torch.randperm(all_x.shape[0])[: n_batches * batch_size]
    for i in range(0, len(indices), batch_size):
        idx = indices[i : i + batch_size]
        x = all_x[idx].to(DEVICE)
        y = all_y[idx].to(DEVICE)
        output = model(x, labels=y)
        losses.append(output.loss.item())
    return sum(losses) / len(losses) if losses else float("inf")


def optimizer_memory(optimizer):
    gpu_bytes = 0
    cpu_bytes = 0
    breakdown = {}
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                b = v.element_size() * v.nelement()
                if v.is_cuda:
                    gpu_bytes += b
                else:
                    cpu_bytes += b
                breakdown[k] = breakdown.get(k, 0) + b
    return (
        gpu_bytes / 1024**2,
        cpu_bytes / 1024**2,
        {k: v / 1024**2 for k, v in breakdown.items()},
    )


def estimate_tflops(n_params, seq_len, batch_size, ms_per_step):
    fwd_flops = 2 * n_params * seq_len * batch_size
    bwd_flops = 4 * n_params * seq_len * batch_size
    total_flops = fwd_flops + bwd_flops
    tflops = total_flops / (ms_per_step / 1000) / 1e12
    return tflops


def train(name, model, optimizer, all_x, all_y, cfg: TrainConfig):
    model.train()

    n_params, n_trainable = count_params(model)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_tflops_peak = 94.81

    print(f"\n{'=' * 140}")
    print(f"  {name}")
    print(
        f"  params: {n_params / 1e6:.2f}M | Llama ({cfg.d_model}d, {cfg.n_layer}L, {cfg.n_head}h)"
        f" | seq={cfg.max_seq_len}, bs={cfg.batch_size}, data={all_x.shape[0]} seqs"
    )
    print(f"{'=' * 140}")
    print(
        f"{'step':>5} {'loss':>8} {'eval':>8} {'gap':>7}"
        f" {'data':>6} {'fwd':>6} {'bwd':>6} {'opt':>6} {'tot':>6}"
        f" {'tok/s':>8} {'TFLOPS':>7} {'MFU%':>5} {'VRAM':>7} {'opt(G/C)':>12} {'grad_n':>8} {'p_norm':>7}"
    )
    print("-" * 140)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    n_data = all_x.shape[0]
    perm = torch.randperm(n_data)
    data_idx = 0

    fwd_acc = bwd_acc = opt_acc = data_acc = 0.0
    tokens_acc = 0
    t0 = time.perf_counter()

    for step in range(1, cfg.max_steps + 1):
        td = time.perf_counter()
        if data_idx + cfg.batch_size > n_data:
            perm = torch.randperm(n_data)
            data_idx = 0
        idx = perm[data_idx : data_idx + cfg.batch_size]
        data_idx += cfg.batch_size
        batch_x = all_x[idx].to(DEVICE, non_blocking=True)
        batch_y = all_y[idx].to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        data_acc += (time.perf_counter() - td) * 1000

        B, T = batch_x.shape
        tokens_acc += B * T

        torch.cuda.synchronize()
        tf = time.perf_counter()
        output = model(batch_x, labels=batch_y)
        loss = output.loss
        torch.cuda.synchronize()
        fwd_acc += (time.perf_counter() - tf) * 1000

        torch.cuda.synchronize()
        tb = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_acc += (time.perf_counter() - tb) * 1000

        torch.cuda.synchronize()
        to = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_acc += (time.perf_counter() - to) * 1000

        if step % cfg.eval_every == 0:
            total_grad_norm = 0.0
            total_param_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.float().norm().item() ** 2
                total_param_norm += p.data.float().norm().item() ** 2
            grad_norm = total_grad_norm**0.5
            param_norm = total_param_norm**0.5
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            opt_gpu, opt_cpu, _ = optimizer_memory(optimizer)

        optimizer.zero_grad()

        if step % cfg.eval_every == 0:
            total_ms = (time.perf_counter() - t0) / cfg.eval_every * 1000
            fwd_avg = fwd_acc / cfg.eval_every
            bwd_avg = bwd_acc / cfg.eval_every
            opt_avg = opt_acc / cfg.eval_every
            data_avg = data_acc / cfg.eval_every
            elapsed = total_ms * cfg.eval_every / 1000
            tok_per_s = tokens_acc / elapsed if elapsed > 0 else 0

            t0 = time.perf_counter()
            fwd_acc = bwd_acc = opt_acc = data_acc = 0.0
            tokens_acc = 0

            eval_loss = evaluate(model, all_x, all_y, n_batches=10, batch_size=16)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            gap = eval_loss - loss.item()

            compute_ms = fwd_avg + bwd_avg
            tflops = estimate_tflops(
                n_params, cfg.max_seq_len, cfg.batch_size, compute_ms
            )
            mfu = (tflops / gpu_tflops_peak) * 100

            print(
                f"{step:>5} {loss.item():>8.4f} {eval_loss:>8.4f} {gap:>7.4f}"
                f" {data_avg:>6.1f} {fwd_avg:>6.1f} {bwd_avg:>6.1f} {opt_avg:>6.1f} {total_ms:>6.1f}"
                f" {tok_per_s:>8.0f} {tflops:>7.1f} {mfu:>5.1f} {vram_mb:>6.0f}MB {opt_gpu:>5.0f}/{opt_cpu:>5.0f}MB {grad_norm:>8.2f} {param_norm:>7.2f}"
            )

    return model


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

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama

    all_x, all_y = preprocess_and_cache(tokenizer, cfg)

    for mod in list(sys.modules.keys()):
        if "grokadamw" in mod.lower():
            del sys.modules[mod]
    sys.path.insert(0, os.path.dirname(__file__))
    from grokadamw import GrokAdamW as NewGrokAdamW
    import grokadamw as _new_ref

    print(f"\n[GrokAdamW from: {_new_ref.__file__}]")

    seed_everything(42)
    apply_liger_kernel_to_llama()
    model = create_model(cfg)
    opt = NewGrokAdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    train(
        "GrokAdamW",
        model,
        opt,
        all_x,
        all_y,
        cfg,
    )
    del model, opt
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    print(f"\n{'=' * 115}")
    print("DONE")
    print(f"{'=' * 115}")


if __name__ == "__main__":
    main()
