from __future__ import annotations

import argparse
import json
import math
import time
import torch

from .data import CharDataset
from .model import TinyGPT, TinyGPTConfig


def evaluate(model, ds, device, batch_size, batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.batch("val", batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.detach()))
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(loss, 20))


def run(mode: str, steps: int, device_name: str, seed: int = 1337):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    ds = CharDataset(block_size=32)
    cfg = TinyGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mode=mode,
        n_qubits=4,
        circuit_layers=2,
    )
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch_size = 4 if mode == "quantum" else 16

    started = time.perf_counter()
    for step in range(steps):
        x, y = ds.batch("train", batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"[{mode}] step={step:4d} train_loss={float(loss.detach()):.4f}")

    elapsed = time.perf_counter() - started
    val_loss, ppl = evaluate(model, ds, device, batch_size)
    seed_ids = ds.encode("quantum ").to(device)
    sample = ds.decode(model.generate(seed_ids, max_new_tokens=60).cpu())
    result = {
        "mode": mode,
        "parameters": model.parameter_count(),
        "val_loss": val_loss,
        "perplexity": ppl,
        "seconds": elapsed,
        "sample": sample,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["classical", "bottleneck", "quantum"], default="classical")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args.mode, args.steps, args.device)


if __name__ == "__main__":
    main()
