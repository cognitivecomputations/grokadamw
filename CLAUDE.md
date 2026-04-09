# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GrokAdamW is a PyTorch optimizer that integrates Grokfast (slow-gradient amplification) into AdamW. Published as the `grokadamw` PyPI package (v0.2.0, Apache 2.0 license).

## Package Structure

- `grokadamw/grokadamw.py` — Optimizer implementation (`GrokAdamW` class extending `torch.optim.Optimizer`)
- `grokadamw/_triton_fused.py` — Triton fused kernel for GPU-accelerated step (optional, auto-detected)
- `grokadamw/benchmark.py` — Benchmark suite (`python -m grokadamw.benchmark`)
- `grokadamw/tests/test_correctness.py` — Correctness and equivalence tests
- `grokadamw/__init__.py` — Exports `GrokAdamW` and `__version__`
- `pyproject.toml` — Build config (setuptools), deps: `torch>=1.0.0`, optional: `triton>=3.0.0`

## Key Architecture Details

The optimizer combines three mechanisms on top of standard AdamW:

1. **Grokfast EMA filter** — Maintains an exponential moving average of gradients (`grok_ema`) per parameter. The filtered gradient is `grad + lamb * grok_ema`, which amplifies slow-varying gradient components.
2. **Adaptive alpha** — The EMA momentum (`alpha`) decays based on a "grokking signal" (a scalar reflecting generalization gap). When no signal functions are provided, it falls back to `(eval_loss - train_loss) / max_loss`.
3. **Layer-wise beta1 decay** — The Adam beta1 is scaled per-parameter-index: `beta1 * (1 - gamma)^i`, so earlier parameters get higher momentum.

### State Storage (`cpu_offload`)

- **`cpu_offload=False` (default):** State tensors (`exp_avg`, `exp_avg_sq`, `grok_ema`) are stored on GPU as float32. Zero CPU↔GPU transfers per step. Uses ~3× param count extra VRAM.
- **`cpu_offload=True`:** States stored on CPU, moved to GPU per step (original v0.1 behavior). For memory-constrained setups.

### Execution Paths

1. **Triton fused kernel** — Activated when `cpu_offload=False` + CUDA available + triton installed. Fuses all 7+ elementwise ops (grok_ema update, grok_grad computation, moment updates, weight decay, parameter update) into a single kernel launch. `grok_grad` computed in registers, never materialized as tensor.
2. **PyTorch fallback** — Used when Triton unavailable or `cpu_offload=True`. Uses explicit `grad.float()` cast (no deprecated autocast).

State tensors are always float32 regardless of param dtype. Gradient clipping is per-parameter. `state_dict`/`load_state_dict` null signal functions on save, restore from defaults on load, and migrate old CPU/non-fp32 states to current config.

## Development

```bash
# Install in editable mode
pip install -e .

# With optional Triton support
pip install -e ".[triton]"

# Run tests
pytest grokadamw/tests/ -v

# Run benchmarks (requires CUDA)
python -m grokadamw.benchmark

# Build for distribution
pip install build
python -m build
```
