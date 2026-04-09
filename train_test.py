import torch
import torch.nn as nn
import time
import sys
import os
import importlib
import importlib.util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TinyTransformer(nn.Module):
    def __init__(self, vocab=256, d_model=256, nhead=4, layers=4, seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, vocab)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(pos)
        h = self.transformer(h)
        return self.head(h)


def make_data(batch, seq_len, vocab):
    x = torch.randint(0, vocab, (batch, seq_len), device=DEVICE)
    return x[:, :-1], x[:, 1:]


def train(name, model, opt_fn, steps=5000, eval_every=500):
    optimizer = opt_fn(model)
    vocab = 256
    seq_len = 128
    batch = 32

    print(f"\n{'=' * 90}")
    print(f"  {name}")
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"{'=' * 90}")
    print(
        f"{'step':>6} {'train_loss':>10} {'eval_loss':>10} {'gen_gap':>8}"
        f" {'fwd_ms':>7} {'bwd_ms':>7} {'opt_ms':>7} {'total_ms':>8}"
    )
    print("-" * 90)

    fwd_acc = 0.0
    bwd_acc = 0.0
    opt_acc = 0.0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        x, y = make_data(batch, seq_len, vocab)

        torch.cuda.synchronize()
        tf = time.perf_counter()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        )
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
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_acc += (time.perf_counter() - to) * 1000

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(10):
                    x, y = make_data(batch, seq_len, vocab)
                    logits = model(x)
                    el = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, vocab), y.reshape(-1)
                    )
                    eval_losses.append(el.item())
                eval_loss = sum(eval_losses) / len(eval_losses)

            total_ms = (time.perf_counter() - t0) / eval_every * 1000
            t0 = time.perf_counter()

            fwd_avg = fwd_acc / eval_every
            bwd_avg = bwd_acc / eval_every
            opt_avg = opt_acc / eval_every
            gap = eval_loss - loss.item()
            print(
                f"{step:>6} {loss.item():>10.4f} {eval_loss:>10.4f} {gap:>8.4f}"
                f" {fwd_avg:>7.2f} {bwd_avg:>7.2f} {opt_avg:>7.2f} {total_ms:>8.1f}"
            )
            fwd_acc = 0.0
            bwd_acc = 0.0
            opt_acc = 0.0

    return model


def main():
    steps = 5000
    seq_len = 128
    vocab = 256

    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")
    print(f"Steps: {steps}")

    torch.manual_seed(42)
    model_old = TinyTransformer(vocab=vocab, seq_len=seq_len).to(DEVICE)
    init_state = {k: v.clone() for k, v in model_old.state_dict().items()}

    torch.manual_seed(42)
    model_new = TinyTransformer(vocab=vocab, seq_len=seq_len).to(DEVICE)
    model_new.load_state_dict({k: v.clone() for k, v in init_state.items()})

    # --- Old GrokAdamW v0.1.2 (pip installed, site-packages) ---
    site_pkg = "/root/train/.venv/lib/python3.12/site-packages"
    old_mod_path = os.path.join(site_pkg, "grokadamw", "grokadamw.py")
    spec = importlib.util.spec_from_file_location("grokadamw_old", old_mod_path)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    OldGrokAdamW = old_module.GrokAdamW
    print(f"\n[Old GrokAdamW loaded from: {old_mod_path}]")

    train(
        "GrokAdamW v0.1.2 OLD (pip, site-packages)",
        model_old,
        lambda m: OldGrokAdamW(m.parameters(), lr=1e-3, weight_decay=1e-2),
        steps=steps,
    )

    # --- New optimized GrokAdamW (local) ---
    for mod in list(sys.modules.keys()):
        if "grokadamw" in mod.lower():
            del sys.modules[mod]
    sys.path.insert(0, os.path.dirname(__file__))
    from grokadamw import GrokAdamW as NewGrokAdamW
    import grokadamw as _gref

    print(f"[New GrokAdamW loaded from: {_gref.__file__}]")

    train(
        "GrokAdamW OPTIMIZED (local, foreach + sync-free clip)",
        model_new,
        lambda m: NewGrokAdamW(m.parameters(), lr=1e-3, weight_decay=1e-2),
        steps=steps,
    )

    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
