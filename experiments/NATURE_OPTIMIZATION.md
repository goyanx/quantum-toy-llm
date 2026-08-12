# Nature-Inspired Quantum Optimization Regimes

## 2026-08-13

This experiment asks whether the best quantum architecture from the previous sweep (`4 qubits`, `2 circuit layers`, data re-uploading, alternating/long-range entanglement) benefits from an optimization regime designed around quantum-trainability concerns rather than a generic learning-rate grid.

The design is inspired by several Nature-family results on variational quantum training:

- *Barren plateaus in variational quantum computing*, Nature Reviews Physics (2025): trainability depends jointly on ansatz, initialization, observable/loss design, noise and optimization landscape, not merely optimizer choice.
- *A Lie algebraic theory of barren plateaus for deep parameterized quantum circuits*, Nature Communications (2024): circuit structure and effective dynamical algebra govern gradient concentration.
- *Engineered dissipation to mitigate barren plateaus*, npj Quantum Information (2024): carefully structured non-unitary/dissipative effects can improve trainability in variational circuits.
- *Noise-induced shallow circuits and the absence of barren plateaus*, Nature Physics (2026): noise can effectively restrict circuits to shallow behavior; avoiding barren plateaus does not itself establish quantum advantage.

The dissipation experiment below is explicitly **inspired by**, not a faithful reproduction of, the open-system method in the cited work.

## Protocol

| Setting | Value |
|---|---:|
| Dataset | built-in character dataset |
| Architecture | quantum re-upload + alternating entanglement |
| Qubits | 4 |
| Circuit layers | 2 |
| Training steps | **500** |
| Batch size | 4 |
| Evaluation batches | 30 |
| Seeds | 1337, 2026, 4242 |
| Device | CPU |
| Parallelism | **4 workers** |

A total of **15 independent runs** were executed: 5 optimization regimes × 3 seeds. Values below are mean ± sample standard deviation.

## Regimes

1. **AdamW baseline** — all parameters train from the start at `lr=0.003`.
2. **Quantum-group cosine** — separate learning rates for classical Transformer (`3e-3`), quantum interface (`2e-3`) and circuit parameters (`7.5e-4`), with cosine decay.
3. **Progressive unfreezing** — circuit frozen for the first 20%, first circuit layer trained from 20–50%, then both layers trained for the final 50%, retaining AdamW at `3e-3`.
4. **Progressive + quantum-group cosine** — combines 2 and 3.
5. **Dissipation-inspired progressive** — regime 4 plus a mild training-only non-unitary state contraction toward `|0...0>`, annealed from `0.03` to zero by 60% of training. This is a regularizer, not a physically exact Markovian channel.

## Results

| Regime | Validation loss | Perplexity | Train time (s) | Mean circuit grad norm |
|---|---:|---:|---:|---:|
| **AdamW baseline** | **2.0301 ± 0.0328** | **7.62 ± 0.25** | 6.39 | 0.0327 ± 0.0091 |
| **Progressive unfreezing** | 2.0321 ± 0.0345 | **7.63 ± 0.26** | **5.35** | 0.0169 ± 0.0007 |
| Quantum-group cosine | 2.3669 ± 0.0292 | 10.67 ± 0.31 | 5.97 | 0.0272 ± 0.0090 |
| Dissipation-inspired progressive | 2.3673 ± 0.0288 | 10.67 ± 0.31 | 5.63 | 0.0124 ± 0.0020 |
| Progressive + quantum-group cosine | 2.3674 ± 0.0296 | 10.67 ± 0.31 | 5.30 | 0.0113 ± 0.0024 |

Exact seed-level measurements are stored in `nature_optimization_raw.csv`.

## Observations

1. **Progressive unfreezing is the only alternative regime that preserved baseline quality.** Its mean perplexity (7.63) is effectively tied with ordinary AdamW (7.62) at this scale.
2. Progressive training reduced the mean quantum-circuit gradient norm by roughly half while retaining predictive quality.
3. The progressive run was around **16% faster** in this implementation because circuit parameters do not require gradients during the early stage.
4. Reducing the circuit learning rate and applying cosine decay was strongly harmful; all three regimes using that schedule clustered near perplexity 10.67.
5. The dissipation-inspired regularizer did **not** rescue the poor grouped/cosine schedule.
6. These results argue against the simplistic rule that quantum parameters should always receive much smaller updates than surrounding classical parameters.

## Updated interpretation

The Nature-inspired lesson that survives this toy experiment is **curriculum over trainable structure**, not generic suppression of quantum gradients.

A useful working hypothesis is:

> Let the classical representation and quantum interface become useful first, then progressively expose more variational circuit freedom while keeping the eventual optimizer aggressive enough to fit the task.

In this small model progressive unfreezing does not improve final perplexity over AdamW, but it reaches essentially the same quality with lower circuit-gradient magnitude and lower runtime.

The engineered-dissipation analogy did not provide a benefit in its current simplified form. That is not a contradiction of the published result: the paper uses specifically designed open-system dynamics, whereas this experiment uses a lightweight state-contraction proxy in a classical statevector simulator.

## Next experiment

1. Compare 10%, 20%, 30% and 40% initial freeze periods.
2. Compare layer-growth schedules: `0→1→2` trainable layers versus `0→2` directly.
3. Keep AdamW at `3e-3`; do not combine this with the failed low-LR cosine schedule.
4. Record full learning curves and gradient variance by circuit layer.
5. Add a classical bottleneck with an analogous progressive-freeze schedule to test whether the benefit is specifically quantum or just a generic curriculum effect.
6. If dissipation is revisited, implement a proper density-matrix amplitude-damping channel rather than the present state-contraction proxy.

## References

- https://www.nature.com/articles/s42254-025-00813-9
- https://www.nature.com/articles/s41467-024-49909-3
- https://www.nature.com/articles/s41534-024-00875-0
- https://www.nature.com/articles/s41567-026-03245-z
