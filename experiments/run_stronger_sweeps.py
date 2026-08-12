from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from quantum_toy_llm.data import CharDataset
from quantum_toy_llm.model import TinyGPT, TinyGPTConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "stronger_sweeps_raw.csv"
SEEDS = [1337, 2026, 4242]
STEPS = 300
BATCH_SIZE = 4
EVAL_BATCHES = 20
LR = 3e-3


def evaluate(model, ds, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = ds.batch("val", BATCH_SIZE, device)
            _, loss = model(x, y)
            losses.append(float(loss.detach()))
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(loss, 20))


def one_run(spec):
    torch.set_num_threads(1)
    seed = spec["seed"]
    torch.manual_seed(seed)
    device = torch.device("cpu")
    ds = CharDataset(block_size=32)
    cfg = TinyGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mode=spec["mode"],
        n_qubits=spec["n_qubits"],
        circuit_layers=spec["circuit_layers"],
    )
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    started = time.perf_counter()
    for _ in range(STEPS):
        x, y = ds.batch("train", BATCH_SIZE, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    elapsed = time.perf_counter() - started
    val_loss, ppl = evaluate(model, ds, device)
    return {
        **spec,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "eval_batches": EVAL_BATCHES,
        "lr": LR,
        "parameters": model.parameter_count(),
        "val_loss": val_loss,
        "perplexity": ppl,
        "seconds": elapsed,
    }


def main():
    specs = []
    for seed in SEEDS:
        specs.append({"sweep": "reference", "mode": "classical", "n_qubits": 4, "circuit_layers": 2, "seed": seed})

    for width in [2, 3, 4, 5]:
        for mode in ["bottleneck", "quantum"]:
            for seed in SEEDS:
                specs.append({"sweep": "width", "mode": mode, "n_qubits": width, "circuit_layers": 2, "seed": seed})

    for depth in [1, 2, 3, 4]:
        for seed in SEEDS:
            specs.append({"sweep": "depth", "mode": "quantum", "n_qubits": 4, "circuit_layers": depth, "seed": seed})

    results = []
    print(f"Running {len(specs)} jobs with max_workers=3", flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one_run, spec): spec for spec in specs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            results.append(row)
            print(
                f"[{i:02d}/{len(specs)}] {row['sweep']:9s} {row['mode']:10s} "
                f"q={row['n_qubits']} L={row['circuit_layers']} seed={row['seed']} "
                f"ppl={row['perplexity']:.3f} t={row['seconds']:.2f}s",
                flush=True,
            )

    results.sort(key=lambda r: (r["sweep"], r["mode"], r["n_qubits"], r["circuit_layers"], r["seed"]))
    fields = [
        "sweep", "mode", "n_qubits", "circuit_layers", "seed", "steps", "batch_size",
        "eval_batches", "lr", "parameters", "val_loss", "perplexity", "seconds"
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
