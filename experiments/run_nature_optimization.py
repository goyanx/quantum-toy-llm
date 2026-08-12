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
OUT = ROOT / "experiments" / "nature_optimization_raw.csv"
SEEDS = [1337, 2026, 4242]
STEPS = 500
BATCH = 4
EVAL = 30
WORKERS = 4

REGIMES = [
    "adamw_baseline",
    "qgroup_cosine",
    "progressive",
    "progressive_qgroup_cosine",
    "dissipation_progressive",
]


def make_model(seed: int):
    torch.manual_seed(seed)
    ds = CharDataset(block_size=32)
    cfg = TinyGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        mode="quantum",
        n_qubits=4,
        circuit_layers=2,
        quantum_reupload=True,
        quantum_topology="alternating",
    )
    return TinyGPT(cfg), ds


def quantum_module(model):
    return model.blocks[0].ffn.quantum


def split_params(model):
    q = quantum_module(model)
    circuit = [q.theta_ry, q.theta_rz]
    if q.input_scale is not None:
        circuit.append(q.input_scale)
    interface = list(q.encoder.parameters()) + list(q.decoder.parameters())
    ids = {id(p) for p in circuit + interface}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return rest, interface, circuit


def set_progressive_trainability(model, step):
    q = quantum_module(model)
    q.theta_ry.requires_grad_(False)
    q.theta_rz.requires_grad_(False)
    if q.input_scale is not None:
        q.input_scale.requires_grad_(False)
    if step >= int(STEPS * 0.20):
        q.theta_ry.requires_grad_(True)
        q.theta_rz.requires_grad_(True)
        if q.input_scale is not None:
            q.input_scale.requires_grad_(True)


def mask_progressive_grads(model, step):
    if not (int(STEPS * 0.20) <= step < int(STEPS * 0.50)):
        return
    q = quantum_module(model)
    for p in [q.theta_ry, q.theta_rz] + ([q.input_scale] if q.input_scale is not None else []):
        if p is not None and p.grad is not None and p.grad.ndim >= 2 and p.grad.shape[0] > 1:
            p.grad[1:].zero_()


def build_optimizer(model, grouped: bool):
    if not grouped:
        return torch.optim.AdamW(model.parameters(), lr=3e-3), None
    rest, interface, circuit = split_params(model)
    opt = torch.optim.AdamW(
        [
            {"params": rest, "lr": 3e-3},
            {"params": interface, "lr": 2e-3},
            {"params": circuit, "lr": 7.5e-4},
        ],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-4)
    return opt, sched


def evaluate(model, ds):
    model.eval()
    quantum_module(model).training_dissipation = 0.0
    losses = []
    with torch.no_grad():
        for _ in range(EVAL):
            x, y = ds.batch("val", BATCH, torch.device("cpu"))
            _, loss = model(x, y)
            losses.append(float(loss))
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(loss, 20))


def run(spec):
    torch.set_num_threads(1)
    model, ds = make_model(spec["seed"])
    regime = spec["regime"]
    progressive = regime in {"progressive", "progressive_qgroup_cosine", "dissipation_progressive"}
    grouped = regime in {"qgroup_cosine", "progressive_qgroup_cosine", "dissipation_progressive"}
    dissipative = regime == "dissipation_progressive"
    opt, sched = build_optimizer(model, grouped)
    q = quantum_module(model)

    grad_norm_sum = 0.0
    grad_norm_sq_sum = 0.0
    grad_samples = 0
    started = time.perf_counter()

    for step in range(STEPS):
        if progressive:
            set_progressive_trainability(model, step)
        else:
            q.theta_ry.requires_grad_(True)
            q.theta_rz.requires_grad_(True)
            if q.input_scale is not None:
                q.input_scale.requires_grad_(True)

        if dissipative:
            frac = max(0.0, 1.0 - step / (STEPS * 0.60))
            q.training_dissipation = 0.03 * frac
        else:
            q.training_dissipation = 0.0

        x, y = ds.batch("train", BATCH, torch.device("cpu"))
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if progressive:
            mask_progressive_grads(model, step)

        g2 = 0.0
        for p in [q.theta_ry, q.theta_rz] + ([q.input_scale] if q.input_scale is not None else []):
            if p is not None and p.grad is not None:
                g2 += float(p.grad.detach().square().sum())
        if g2 > 0:
            gn = math.sqrt(g2)
            grad_norm_sum += gn
            grad_norm_sq_sum += gn * gn
            grad_samples += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()

    seconds = time.perf_counter() - started
    val_loss, ppl = evaluate(model, ds)
    mean_g = grad_norm_sum / max(1, grad_samples)
    var_g = grad_norm_sq_sum / max(1, grad_samples) - mean_g * mean_g
    return {
        "regime": regime,
        "seed": spec["seed"],
        "parameters": model.parameter_count(),
        "val_loss": val_loss,
        "perplexity": ppl,
        "seconds": seconds,
        "circuit_grad_norm_mean": mean_g,
        "circuit_grad_norm_std": math.sqrt(max(0.0, var_g)),
    }


def main():
    specs = [{"regime": r, "seed": s} for r in REGIMES for s in SEEDS]
    rows = []
    print(f"Running {len(specs)} jobs with max_workers={WORKERS}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run, s): s for s in specs}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            rows.append(r)
            print(
                f"[{i:02}/{len(specs)}] {r['regime']:28s} seed={r['seed']} "
                f"ppl={r['perplexity']:.3f} grad={r['circuit_grad_norm_mean']:.4g} t={r['seconds']:.2f}s",
                flush=True,
            )
    rows.sort(key=lambda r: (r["regime"], r["seed"]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
