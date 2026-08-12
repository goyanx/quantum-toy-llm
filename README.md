# Quantum Toy LLM

A deliberately small research playground for the question:

> Can a parameterized quantum circuit replace a relatively large feed-forward sublayer in a Transformer while using fewer trainable parameters?

This repo implements three otherwise-matched character-level TinyGPT models:

1. **Classical** — standard Transformer FFN: `d_model -> 4*d_model -> d_model`.
2. **Bottleneck** — classical control: `d_model -> n_qubits -> d_model`.
3. **Quantum** — `d_model -> n_qubits -> differentiable statevector circuit -> n_qubits -> d_model`.

The quantum path is simulated directly in PyTorch, so no quantum SDK or hardware is required. Autograd trains the circuit parameters end-to-end with the Transformer.

## What this does — and does not — test

It tests **parameter compression and learnability** of a tiny quantum bottleneck. It does **not** demonstrate quantum speedup. Statevector simulation costs grow exponentially as `2^n_qubits`, and classical-to-quantum data loading plus measurement can erase theoretical advantages on real hardware.

## Circuit

For each token representation:

```text
hidden state (d_model)
        |
   Linear projection
        |
   n_qubit angles
        |
   RY data encoding
        |
 [ trainable RY/RZ ] x L
 [ ring of CNOTs   ]
        |
  <Z0> ... <Zn-1>
        |
   Linear projection
        |
hidden state (d_model)
```

The trainable circuit itself has only `2 * n_qubits * circuit_layers` rotation parameters. The encoder/decoder add classical interface parameters.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
```

Train one model:

```bash
qtllm-train --mode classical  --steps 100 --device cpu
qtllm-train --mode bottleneck --steps 100 --device cpu
qtllm-train --mode quantum    --steps 100 --device cpu
```

Run a matched comparison:

```bash
qtllm-compare --steps 100 --device cpu
```

For CUDA, pass `--device cuda`. For this toy statevector simulator, small CPU runs may be simpler; GPU becomes more useful as the batch/circuit workload grows.

## Metrics reported

- total trainable parameters
- validation cross-entropy
- perplexity
- wall-clock training time
- generated text sample

The primary experiment is whether the quantum model can retain acceptable validation loss/perplexity while using fewer parameters than the classical FFN model.

## Suggested experiments

- Sweep `n_qubits`: 2, 3, 4, 5, 6.
- Sweep circuit depth: 1–6 layers.
- Compare against a **classical low-rank bottleneck** with the same latent width. This is essential: fewer parameters alone do not prove that quantum structure is useful.
- Replace the FFN in only one Transformer block versus every block.
- Add parameter-shift gradients and compare them with simulator autograd.
- Add noisy measurement / finite-shot simulation.
- Later, port the same circuit to PennyLane or Qiskit and run on actual quantum backends.

## Scientific caution

A useful result is not simply "the quantum model has fewer parameters." The important comparison is:

```text
full classical FFN
vs.
classical bottleneck with the same parameter budget
vs.
quantum bottleneck
```

If the quantum version consistently beats the matched classical bottleneck on held-out data, that is a much more interesting signal.

## License

MIT
