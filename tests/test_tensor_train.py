import torch
from quantum_toy_llm.tensor_train import TTLinear, TensorTrainFFN


def test_ttlinear_shape_and_backward():
    layer = TTLinear(32, rank=2)
    x = torch.randn(3, 5, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert all(c.grad is not None for c in layer.cores)


def test_ttffn_parameter_compression():
    ffn = TensorTrainFFN(32, rank=2)
    params = sum(p.numel() for p in ffn.parameters())
    dense_two_layer = 2 * (32 * 32 + 32)
    assert params < dense_two_layer
