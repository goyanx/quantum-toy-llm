# Curriculum Ablation: Progressive vs Direct Unfreezing

## 2026-08-13

This experiment follows the Nature-inspired optimization study and asks whether the apparent benefit of progressive quantum training is specific to the quantum circuit or simply a generic effect of delaying a small latent transform.

## Protocol

- Dataset: built-in character corpus
- TinyGPT: `d_model=32`, 1 transformer block, 4 heads
- Quantum architecture: 4-qubit, 2-layer, data re-uploading, alternating entanglement
- Optimizer: AdamW, learning rate `0.003`
- Training: **500 steps**
- Evaluation: 30 validation batches
- Seeds: **1337, 2026, 4242**
- Parallelism: **maximum 4 local workers**
- Values: mean ± sample standard deviation over 3 seeds

Two quantum schedules were tested:

- **gradual `0→1→2`**: freeze all circuit parameters initially, then train the first circuit layer, then both layers.
- **direct `0→2`**: freeze all circuit parameters initially, then unfreeze both layers at once.

Freeze fractions of 10%, 20%, 30%, and 40% were tested.

## Quantum curriculum results

| Schedule | Initial freeze | Perplexity | Train time (s) | Mean circuit grad norm |
|---|---:|---:|---:|---:|
| baseline | 0% | 7.618 ± 0.250 | 5.30 | 0.0327 |
| direct | 10% | 7.638 ± 0.264 | 5.27 | 0.0177 |
| direct | 20% | 7.643 ± 0.262 | 6.07 | 0.0184 |
| direct | 30% | 7.643 ± 0.265 | 5.13 | 0.0189 |
| direct | 40% | 7.628 ± 0.265 | 5.04 | 0.0194 |
| gradual | 10% | 7.626 ± 0.261 | 5.17 | 0.0163 |
| gradual | 20% | 7.634 ± 0.263 | 5.35 | 0.0168 |
| gradual | 30% | 7.625 ± 0.271 | 5.34 | 0.0174 |
| gradual | 40% | 7.626 ± 0.269 | 5.00 | 0.0182 |

### Quantum observations

1. The fully trained AdamW baseline remains best at **7.618 ± 0.250 perplexity**.
2. Every delayed-unfreezing schedule is extremely close to baseline. The best curriculum, gradual 30%, reached **7.625 ± 0.271**.
3. Even freezing the circuit for 40% of training changed perplexity by only about **0.10%**.
4. Progressive schedules roughly halve the mean circuit gradient norm versus baseline, but do not improve final validation quality.
5. `0→1→2` gradual unfreezing does not materially outperform `0→2` direct unfreezing.

## First classical control: freezing the whole narrow bottleneck

A direct translation of the schedule to the original classical bottleneck freezes its narrow FFN layers. This hurts quality substantially:

| Schedule | Freeze | Perplexity |
|---|---:|---:|
| baseline | 0% | 7.375 ± 0.516 |
| direct | 20% | 7.834 ± 0.337 |
| direct | 40% | 7.864 ± 0.299 |
| gradual | 20% | 7.833 ± 0.323 |
| gradual | 40% | 7.870 ± 0.301 |

However, this control is **too harsh**: the quantum schedule keeps the quantum encoder and decoder trainable while freezing only the internal circuit. Freezing the entire classical bottleneck also freezes the analogous input/output transforms.

## Stricter classical latent-layer control

To make the comparison fairer, a parameter-matched classical model was added:

```text
32 → 4 encoder
     ↓
   4 → 4 trainable latent transform
     ↓
4 → 32 decoder
```

Total parameters: **6,584**, very close to the quantum model's **6,588**.

Only the internal `4→4` latent transform is frozen, while encoder/decoder remain trainable—analogous to freezing the quantum circuit while leaving its interface trainable.

| Classical latent schedule | Freeze | Perplexity | Train time (s) |
|---|---:|---:|---:|
| baseline | 0% | 7.580 ± 0.607 | 1.08 |
| direct | 20% | 7.584 ± 0.625 | 0.99 |
| direct | 40% | 7.583 ± 0.631 | 1.01 |
| gradual | 20% | 7.584 ± 0.625 | 0.97 |
| gradual | 40% | 7.583 ± 0.631 | 0.97 |

### Classical latent-control observations

1. The classical latent baseline reached **7.580 ± 0.607**, slightly better on mean than the quantum baseline.
2. Freezing the classical latent transform for 20% or 40% barely changed final quality: both remain around **7.58 perplexity**.
3. Therefore, **tolerance to delayed latent-parameter training is not quantum-specific** in this toy task.
4. The earlier whole-bottleneck degradation was caused largely by freezing the classical encoder/decoder along with the latent transform.

## Updated conclusion

The progressive quantum curriculum remains useful computationally, because it reduces circuit-gradient activity and can skip a substantial fraction of expensive circuit updates without harming final perplexity much. But this experiment does **not** support the stronger claim that progressive unfreezing gives a uniquely quantum optimization advantage.

The cleaner result is:

> Small latent transforms—quantum or classical—can be introduced late in training while the surrounding transformer/interface learns first, with little loss in final quality.

For the quantum model this may still matter practically because circuit-gradient evaluation is the expensive component on real quantum hardware or parameter-shift simulation.

## Next useful experiment

The next test should measure **gradient-evaluation cost explicitly** rather than only wall-clock statevector time:

1. Parameter-shift gradient counts for quantum circuit parameters.
2. Progressive training versus full training under a fixed quantum-evaluation budget.
3. Compare final perplexity at equal numbers of circuit evaluations, not equal training steps.
4. Repeat with finite-shot measurements/noise.

That would test whether curriculum training produces a genuine **quantum-resource efficiency advantage**, even if its final perplexity is not better than a classical latent control.
