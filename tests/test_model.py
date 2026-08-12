import torch
from quantum_toy_llm.model import TinyGPT, TinyGPTConfig


def test_quantum_model_has_fewer_parameters_than_classical_model():
    base = dict(vocab_size=32, block_size=16, d_model=32, n_heads=4, n_layers=1)
    classical = TinyGPT(TinyGPTConfig(**base, mode="classical"))
    bottleneck = TinyGPT(TinyGPTConfig(**base, mode="bottleneck", n_qubits=4))
    quantum = TinyGPT(TinyGPTConfig(**base, mode="quantum", n_qubits=4, circuit_layers=2))
    assert bottleneck.parameter_count() < classical.parameter_count()
    assert quantum.parameter_count() < classical.parameter_count()


def test_forward_loss():
    cfg = TinyGPTConfig(vocab_size=20, block_size=8, d_model=16, n_heads=4, n_layers=1, mode="quantum", n_qubits=3, circuit_layers=1)
    model = TinyGPT(cfg)
    x = torch.randint(0, 20, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 20)
    assert loss is not None and torch.isfinite(loss)
