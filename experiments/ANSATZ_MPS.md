# Quantum Ansatz and Tensor-Train/MPS Compression Experiment

## 2026-08-12

This experiment follows the stronger multi-seed sweeps and tests whether changing the **structure** of the compressed transformation improves on the original quantum bottleneck. It also adds a quantum-inspired tensor-train/MPS-style control.

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
| Training steps | 300 |
| Batch size | 4 |
| Evaluation batches | 20 |
| Seeds | 1337, 2026, 4242 |
| Device | CPU |
| Parallelism | **4 local Python workers** |

There were **21 independent runs**: 7 structures × 3 random seeds. Values below are mean ± sample standard deviation.

Raw per-run results are in `ansatz_mps_raw.csv`. The reproducible runner is `run_ansatz_mps_sweep.py`.

## Structures tested

1. `bottleneck4` — classical `32 -> 4 -> 32` control.
2. `quantum_base` — original 4-qubit, 2-layer ring-entangled circuit.
3. `quantum_reupload_ring` — re-injects the encoded input before every quantum layer, retaining ring CNOT entanglement.
4. `quantum_reupload_alt` — data re-uploading plus alternating nearest-neighbor / longer-range entanglement.
5. `mps_r2` — tensor-train/MPS-style two-layer FFN with TT rank 2.
6. `mps_r4` — TT rank 4.
7. `mps_r8` — TT rank 8.

The tensor-train layer factorizes a 32×32 matrix into three TT cores using mode sizes `(4, 4, 2)`. It is reconstructed densely in this toy implementation, so the experiment measures parameterization quality rather than optimized TT inference speed.

## Results

| Structure | Parameters | Validation loss | Perplexity | Train time (s) |
|---|---:|---:|---:|---:|
| Classical bottleneck width 4 | 6,564 | **2.3114 ± 0.0360** | **10.09 ± 0.37** | 0.74 ± 0.06 |
| Original quantum circuit | 6,580 | 2.3688 ± 0.0112 | 10.69 ± 0.12 | 2.86 ± 0.20 |
| Quantum + data re-upload, ring | 6,588 | 2.3773 ± 0.0230 | 10.78 ± 0.25 | 3.44 ± 0.14 |
| Quantum + data re-upload, alternating | 6,588 | **2.3597 ± 0.0311** | **10.59 ± 0.33** | 3.44 ± 0.16 |
| TT/MPS rank 2 | 6,544 | 2.3941 ± 0.0571 | 10.97 ± 0.64 | 0.82 ± 0.10 |
| TT/MPS rank 4 | 7,008 | 2.3863 ± 0.0532 | 10.88 ± 0.59 | 0.78 ± 0.08 |
| TT/MPS rank 8 | 8,704 | 2.3896 ± 0.0574 | 10.92 ± 0.63 | 0.71 ± 0.02 |

## Findings

### 1. Data re-uploading helps only with the better entangling topology

The alternating re-upload circuit improves mean perplexity from **10.69 to 10.59** relative to the original quantum circuit, about a **0.9% improvement**. Ring re-uploading is worse at **10.78**.

This suggests that re-uploading alone is not sufficient; how information is mixed between qubits matters.

### 2. The improvement is not enough to beat the classical bottleneck

The matched-width classical bottleneck remains best at **10.09 ± 0.37 perplexity**. The best quantum variant is about **5% worse in mean perplexity** while taking about **4.6×** as long to train in this statevector simulator.

There is therefore still **no empirical quantum advantage** in this setup.

### 3. The current TT/MPS parameterization is compact but not competitive

TT rank 2 has **6,544 total model parameters**, slightly fewer than the classical bottleneck and quantum models, but reaches only **10.97 ± 0.64 perplexity**.

Increasing TT rank to 4 or 8 does not produce a monotonic improvement. Rank 4 reaches 10.88 and rank 8 reaches 10.92. The extra parameters are not paying off under this training setup.

### 4. Structure matters more than parameter count alone

All compressed models live in a fairly narrow parameter range, yet their performance differs. The classical bottleneck remains difficult to beat despite its very simple transformation. This is an important control: compression itself is easy; **useful compression is the harder target**.

## Updated conclusion

The best new result is the **4-qubit, 2-layer data re-uploading circuit with alternating entanglement**. It improves on the original quantum ansatz, but only modestly and not enough to beat the classical control.

The TT/MPS experiment is also informative: a quantum-inspired tensor factorization is not automatically superior simply because it has a compact representation.

Current ranking by mean perplexity:

1. Classical bottleneck width 4 — **10.09**
2. Quantum re-upload + alternating entanglement — **10.59**
3. Original quantum circuit — **10.69**
4. Quantum re-upload + ring — **10.78**
5. TT/MPS rank 4 — **10.88**
6. TT/MPS rank 8 — **10.92**
7. TT/MPS rank 2 — **10.97**

## Recommended next experiment

The next useful test should focus on optimization and representation rather than more raw circuit depth:

- learning-rate sweep separately for classical, quantum, and TT/MPS models;
- trainable input scaling / data re-uploading depth sweep;
- stronger entanglement patterns (brick-wall and all-to-all for only 4 qubits);
- parameter-matched TT ranks rather than rank-only comparison;
- learning curves and area-under-learning-curve metrics;
- repeat the best candidates on TinyShakespeare or a small TinyStories sample.

These results remain exploratory and should not be presented as evidence of quantum speedup or quantum advantage.
