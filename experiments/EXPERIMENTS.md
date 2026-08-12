# Experiment Log

## 2026-08-12 — Initial Hybrid Quantum TinyGPT Sweeps

This log records the first controlled sweeps for `quantum-toy-llm`.

The goal is not to demonstrate quantum speedup. The immediate question is narrower:

> Can the simulated quantum bottleneck provide a useful parameter/quality tradeoff compared with both the full Transformer FFN and a classical bottleneck with a similar parameter budget?

## Common setup

All runs used the current repository implementation with the following fixed settings:

| Setting | Value |
|---|---:|
| Dataset | built-in character dataset |
| Block size | 32 |
| `d_model` | 32 |
| Attention heads | 4 |
| Transformer layers | 1 |
| Optimizer | AdamW |
| Learning rate | 0.003 |
| Training steps | 30 |
| Batch size | 4 |
| Evaluation batches | 10 |
| Seed | 1337 |
| Device | CPU |

These are deliberately short smoke-scale experiments. They are useful for checking trends and implementation behavior, but they are **not statistically robust**. A follow-up should use more steps and multiple seeds.

## Reference: full classical FFN

The full classical model was trained under the same 30-step, batch-size-4 protocol for context.

| Model | Parameters | Validation loss | Perplexity | Train time (s) |
|---|---:|---:|---:|---:|
| Full classical FFN | 14,624 | 3.4993 | 33.09 | 0.075 |

The compressed models below use roughly 6.4K–6.6K parameters, or about 55% fewer total parameters than this baseline, but at this very short training horizon they remain far behind the full FFN in perplexity.

---

# Sweep 1 — Qubit width

## Question

How does bottleneck width affect the quantum model, and does the quantum transformation outperform a classical bottleneck with approximately the same width and parameter budget?

## Fixed values

- circuit layers: 2
- qubit widths tested: 2, 3, 4, 5
- each quantum configuration is paired with a classical bottleneck of the same latent width

## Results

| Width / qubits | Model | Parameters | Validation loss | Perplexity | Train time (s) |
|---:|---|---:|---:|---:|---:|
| 2 | Classical bottleneck | 6,434 | 7.0692 | 1175.18 | 0.073 |
| 2 | Quantum bottleneck | 6,442 | 6.5474 | **697.44** | 0.289 |
| 3 | Classical bottleneck | 6,499 | 6.2009 | 493.17 | 0.115 |
| 3 | Quantum bottleneck | 6,511 | 6.1030 | **447.21** | 0.350 |
| 4 | Classical bottleneck | 6,564 | 6.7104 | 820.87 | 0.087 |
| 4 | Quantum bottleneck | 6,580 | 6.4759 | **649.27** | 0.430 |
| 5 | Classical bottleneck | 6,629 | 5.4948 | **243.43** | 0.094 |
| 5 | Quantum bottleneck | 6,649 | 6.7822 | 881.98 | 0.589 |

## Observations

1. At widths 2, 3, and 4, the quantum bottleneck achieved lower validation loss and perplexity than the same-width classical bottleneck in this particular single-seed short run.
2. The strongest quantum result in this sweep was 3 qubits: perplexity 447.21 versus 493.17 for its matched classical bottleneck.
3. At width 5 the pattern reversed sharply: the classical bottleneck reached perplexity 243.43 while the quantum model reached 881.98.
4. Quantum simulation time increased with qubit count, as expected from statevector simulation. The 5-qubit quantum run took about 6.2x the training time of its classical bottleneck counterpart.
5. None of the compressed models approached the full classical FFN's perplexity of 33.09 after only 30 steps.

## Interpretation

There is **no evidence yet of a general quantum advantage**. The 2–4 qubit results are enough to justify further testing because the quantum circuit beat its matched bottleneck controls under those settings. However, the 5-qubit reversal shows that simply adding qubits does not monotonically improve learning.

Possible causes include optimization sensitivity, circuit expressivity that is poorly matched to the task, the fixed learning rate, random initialization, or the very short training horizon.

---

# Sweep 2 — Quantum circuit depth

## Question

At fixed 4-qubit width, does increasing the number of trainable quantum layers improve validation quality enough to justify the additional parameters and simulation cost?

## Fixed values

- qubits: 4
- circuit depths tested: 1, 2, 3, 4
- all other settings unchanged

## Results

| Circuit layers | Parameters | Validation loss | Perplexity | Train time (s) |
|---:|---:|---:|---:|---:|
| 1 | 6,572 | **6.3305** | **561.42** | 0.271 |
| 2 | 6,580 | 6.4759 | 649.27 | 0.397 |
| 3 | 6,588 | 6.8284 | 923.71 | 0.482 |
| 4 | 6,596 | 6.5877 | 726.11 | 0.592 |

## Observations

1. One circuit layer produced the best validation loss and perplexity in this sweep.
2. Increasing depth from 1 to 3 progressively worsened perplexity despite adding trainable circuit parameters.
3. Four layers recovered somewhat relative to three layers, but remained substantially worse than one layer.
4. Simulation time increased almost linearly over this small range: 0.271 s at one layer versus 0.592 s at four layers.
5. Under this training budget, extra circuit depth did not pay for itself.

## Interpretation

For this toy setup, **shallower is currently better**. Deeper circuits may need different initialization, learning rates, more training steps, or alternative entangling patterns. This may also be an early sign of difficult optimization as circuit depth grows, but this experiment is too small to attribute the result to barren plateaus or any specifically quantum training pathology.

---

# Current conclusion

The first two sweeps give a useful direction for the next experiment rather than a final result:

- Keep **3–4 qubits** as the most interesting width range for now.
- Prefer **1–2 circuit layers** until deeper circuits demonstrate a measurable benefit.
- Always retain the same-width classical bottleneck as the critical control.
- The full FFN still performs much better at this short training horizon, so parameter reduction is currently being purchased with a large quality loss.

The most important positive signal is that, in three of four matched-width cases, the quantum bottleneck did better than the comparable narrow classical bottleneck. The most important negative signal is that this was inconsistent and did not survive the 5-qubit setting.

## Next experiments

A stronger follow-up should:

1. Run 3–5 random seeds per configuration and report mean ± standard deviation.
2. Increase training from 30 steps to at least 300–1,000 steps.
3. Focus first on 3 and 4 qubits with 1 and 2 circuit layers.
4. Compare learning curves, not only final perplexity.
5. Add parameter-matched classical controls rather than matching width alone.
6. Optionally test a different entangling topology and finite-shot/noisy measurements.

Until those runs are completed, these results should be treated as exploratory evidence only.
