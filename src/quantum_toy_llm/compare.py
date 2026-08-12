from __future__ import annotations

import argparse
from .train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    classical = run("classical", args.steps, args.device, seed=1337)
    bottleneck = run("bottleneck", args.steps, args.device, seed=1337)
    quantum = run("quantum", args.steps, args.device, seed=1337)

    q_reduction = 100 * (1 - quantum["parameters"] / classical["parameters"])
    b_reduction = 100 * (1 - bottleneck["parameters"] / classical["parameters"])
    print("\n=== comparison ===")
    print(f"full classical params : {classical['parameters']:,}")
    print(f"bottleneck params     : {bottleneck['parameters']:,} ({b_reduction:.2f}% reduction)")
    print(f"quantum params        : {quantum['parameters']:,} ({q_reduction:.2f}% reduction)")
    print(f"full classical ppl    : {classical['perplexity']:.3f}")
    print(f"bottleneck ppl        : {bottleneck['perplexity']:.3f}")
    print(f"quantum ppl           : {quantum['perplexity']:.3f}")


if __name__ == "__main__":
    main()
