import torch
from quantum_toy_llm.quantum import QuantumBottleneck


def test_quantum_bottleneck_shape_and_grad():
    layer = QuantumBottleneck(d_model=8, n_qubits=3, circuit_layers=1)
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert layer.theta_ry.grad is not None
