from __future__ import annotations

import math
import torch
from torch import nn


def _apply_single_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a 2x2 gate to one qubit of a batched statevector."""
    batch = state.shape[0]
    shaped = state.view(batch, *([2] * n_qubits))
    axis = qubit + 1
    shaped = shaped.movedim(axis, -1)
    out = torch.einsum("bij,b...j->b...i", gate, shaped)
    return out.movedim(-1, axis).reshape(batch, -1)


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    gate = torch.zeros(theta.shape[0], 2, 2, dtype=torch.complex64, device=theta.device)
    gate[:, 0, 0] = c
    gate[:, 0, 1] = -s
    gate[:, 1, 0] = s
    gate[:, 1, 1] = c
    return gate


def _rz(theta: torch.Tensor) -> torch.Tensor:
    gate = torch.zeros(theta.shape[0], 2, 2, dtype=torch.complex64, device=theta.device)
    gate[:, 0, 0] = torch.exp(-0.5j * theta)
    gate[:, 1, 1] = torch.exp(0.5j * theta)
    return gate


def _cnot_permutation(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(2**n_qubits, device=device)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    flip = torch.where((idx & control_mask) != 0, target_mask, 0)
    return idx ^ flip


class QuantumBottleneck(nn.Module):
    """Small differentiable statevector circuit for tiny hybrid experiments."""

    def __init__(self, d_model: int, n_qubits: int = 4, circuit_layers: int = 2):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.circuit_layers = circuit_layers
        self.encoder = nn.Linear(d_model, n_qubits)
        self.theta_ry = nn.Parameter(torch.zeros(circuit_layers, n_qubits))
        self.theta_rz = nn.Parameter(torch.zeros(circuit_layers, n_qubits))
        nn.init.normal_(self.theta_ry, std=0.05)
        nn.init.normal_(self.theta_rz, std=0.05)
        self.decoder = nn.Linear(n_qubits, d_model)

    @property
    def circuit_parameter_count(self) -> int:
        return self.theta_ry.numel() + self.theta_rz.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        angles = math.pi * torch.tanh(self.encoder(flat))
        batch = flat.shape[0]

        state = torch.zeros(batch, 2**self.n_qubits, dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j

        for q in range(self.n_qubits):
            state = _apply_single_qubit_gate(state, _ry(angles[:, q]), q, self.n_qubits)

        for layer in range(self.circuit_layers):
            for q in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(self.theta_ry[layer, q].expand(batch)), q, self.n_qubits)
                state = _apply_single_qubit_gate(state, _rz(self.theta_rz[layer, q].expand(batch)), q, self.n_qubits)
            for q in range(self.n_qubits):
                target = (q + 1) % self.n_qubits
                perm = _cnot_permutation(self.n_qubits, q, target, x.device)
                state = state[:, perm]

        probs = state.abs().square()
        basis = torch.arange(2**self.n_qubits, device=x.device)
        expectations = []
        for q in range(self.n_qubits):
            mask = 1 << (self.n_qubits - 1 - q)
            signs = torch.where((basis & mask) == 0, 1.0, -1.0)
            expectations.append((probs * signs).sum(dim=-1))
        z = torch.stack(expectations, dim=-1)
        out = self.decoder(z)
        return out.view(*original_shape)
