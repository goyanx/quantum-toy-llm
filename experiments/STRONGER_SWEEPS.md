# Stronger Multi-Seed Sweeps

## 2026-08-12

This follow-up repeats the initial width and circuit-depth sweeps with a stronger protocol to test whether the 30-step, single-seed patterns survive longer training and seed variation.

## Protocol

| Setting | Value |
|---|---:|
| Dataset | built-in character dataset |
| Block size | 32 |
| `d_model` | 32 |
| Attention heads | 4 |
| Transformer layers | 1 |
| Optimizer | AdamW |
| Learning rate | 0.003 |
| Training steps | **300** |
| Batch size | **4 for every model** |
| Evaluation batches | **20** |
| Seeds | **1337, 2026, 4242** |
| Device | CPU |
| Parallelism | **maximum 3 workers** |

A total of **39 independent training runs** were executed: 3 full-classical reference runs, 24 width-sweep runs, and 12 depth-sweep runs. Values below are mean ± sample standard deviation across three seeds.

Exact per-run values are in `stronger_sweeps_raw.csv`. The reproducible runner is `run_stronger_sweeps.py`.

## Full classical reference

| Model | Parameters | Validation loss | Perplexity | Train time (s) |
|---|---:|---:|---:|---:|
| Full classical FFN | 14,624 | 1.7289 ± 0.0219 | **5.64 ± 0.12** | 0.59 ± 0.01 |

The full FFN remains substantially better in predictive quality than every compressed model after 300 steps.

# Sweep 1 — Qubit / bottleneck width

Circuit depth is fixed at 2 layers. Each quantum model is compared against a classical bottleneck with the same latent width.

| Width | Model | Parameters | Validation loss | Perplexity | Train time (s) |
|---:|---|---:|---:|---:|---:|
| 2 | Classical bottleneck | 6,434 | **2.3235 ± 0.0610** | **10.22 ± 0.62** | 0.57 ± 0.02 |
| 2 | Quantum bottleneck | 6,442 | 2.3543 ± 0.0313 | 10.53 ± 0.33 | 1.52 ± 0.05 |
| 3 | Classical bottleneck | 6,499 | **2.3497 ± 0.0408** | **10.49 ± 0.43** | 0.57 ± 0.01 |
| 3 | Quantum bottleneck | 6,511 | 2.3921 ± 0.0389 | 10.94 ± 0.43 | 1.92 ± 0.04 |
| 4 | Classical bottleneck | 6,564 | **2.3114 ± 0.0360** | **10.09 ± 0.37** | 0.56 ± 0.02 |
| 4 | Quantum bottleneck | 6,580 | 2.3688 ± 0.0112 | 10.69 ± 0.12 | 2.62 ± 0.04 |
| 5 | Classical bottleneck | 6,629 | **2.2846 ± 0.0520** | **9.83 ± 0.52** | 0.57 ± 0.02 |
| 5 | Quantum bottleneck | 6,649 | 2.3793 ± 0.0350 | 10.80 ± 0.38 | 3.68 ± 0.13 |

## Width-sweep observations

1. The apparent quantum advantage seen at widths 2, 3, and 4 in the original 30-step single-seed experiment **did not survive** the stronger protocol.
2. The classical bottleneck has lower mean validation loss and perplexity at **all four tested widths**.
3. The best compressed model is the 5-wide classical bottleneck at **9.83 ± 0.52 perplexity**.
4. The best quantum width result is the 2-qubit model at **10.53 ± 0.33 perplexity**.
5. Statevector simulation cost rises sharply with width. At width 5 the quantum run takes about **6.5x** the training time of the matched classical bottleneck while also producing worse perplexity.

### Interpretation

Under longer multi-seed training, there is currently **no evidence that this quantum circuit provides a better width-matched compression transform than a simple classical bottleneck**. The initial 30-step result was likely dominated by initialization and early-optimization effects.

# Sweep 2 — Quantum circuit depth

Qubit width is fixed at 4. Circuit depths 1 through 4 are compared across the same three seeds.

| Circuit layers | Parameters | Validation loss | Perplexity | Train time (s) |
|---:|---:|---:|---:|---:|
| 1 | 6,572 | 2.3597 ± 0.0616 | 10.60 ± 0.66 | 1.90 ± 0.08 |
| 2 | 6,580 | 2.3688 ± 0.0112 | 10.69 ± 0.12 | 3.07 ± 0.07 |
| 3 | 6,588 | 2.3464 ± 0.0403 | 10.45 ± 0.43 | 4.08 ± 0.09 |
| 4 | 6,596 | **2.3417 ± 0.0170** | **10.40 ± 0.18** | 4.97 ± 0.14 |

## Depth-sweep observations

1. The original 30-step result suggested one circuit layer was best. That finding **does not hold after 300 steps and three seeds**.
2. Four layers produce the best mean validation loss and perplexity, with three layers close behind.
3. The quality differences among depths are modest: mean perplexity ranges from 10.40 to 10.69.
4. Runtime rises strongly with depth: four layers take roughly **2.6x** as long as one layer for only about a **1.9% reduction in mean perplexity**.

### Interpretation

Longer training changes the optimization picture. Deeper circuits can catch up to and slightly outperform shallow circuits, but the gain is small relative to simulation cost.

# Updated conclusion

The stronger experiment changes the earlier interpretation substantially:

- **The matched classical bottleneck wins at every tested width.**
- The earlier quantum wins at widths 2–4 were not robust to longer training and multiple seeds.
- The full classical FFN remains much better in absolute quality: 5.64 perplexity versus roughly 9.8–10.9 for compressed models.
- Deeper quantum circuits improve slightly after sufficient training, but runtime cost grows much faster than the quality benefit.
- The current circuit is a valid differentiable compressed transformation, but it has **not demonstrated an empirical advantage over the simpler classical control**.

## Next experiments

1. Parameter-match the quantum and classical models more strictly rather than matching only bottleneck width.
2. Test a different encoding/ansatz: data re-uploading, alternate entanglement topologies, and trainable input scaling.
3. Run a learning-rate sweep because the two architectures may prefer different optimizer settings.
4. Record learning curves instead of only final metrics.
5. Repeat on a less repetitive corpus such as TinyShakespeare or a small TinyStories subset.
6. Add an MPS/tensor-train FFN as a quantum-inspired compression control.

The present evidence argues against claiming a quantum advantage, while providing a clean baseline for alternative quantum and quantum-inspired structures.
