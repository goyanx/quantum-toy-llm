from __future__ import annotations

import math
import torch
from torch import nn


def _apply_single_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
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


def _entangle_pairs(n_qubits: int, topology: str, layer: int):
    if topology == "ring":
        return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    if topology == "alternating":
        # Alternate nearest-neighbor and longer-range couplings.
        if layer % 2 == 0:
            return [(q, q + 1) for q in range(0, n_qubits - 1, 2)] + [(q, q + 1) for q in range(1, n_qubits - 1, 2)]
        step = max(2, n_qubits // 2)
        pairs = []
        for q in range(n_qubits):
            t = (q + step) % n_qubits
            if q < t:
                pairs.append((q, t))
        return pairs or [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    raise ValueError(f"unknown entanglement topology: {topology}")


class QuantumBottleneck(nn.Module):
    """Differentiable statevector circuit for tiny hybrid experiments."""

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        circuit_layers: int = 2,
        *,
        reupload: bool = False,
        topology: str = "ring",
    ):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.circuit_layers = circuit_layers
        self.reupload = reupload
        self.topology = topology
        # Optional training-time non-unitary contraction used only in the
        # Nature-inspired optimization experiment. Keep at 0.0 for the
        # standard model. The experiment anneals this back to zero.
        self.training_dissipation = 0.0
        self.encoder = nn.Linear(d_model, n_qubits)
        self.theta_ry = nn.Parameter(torch.zeros(circuit_layers, n_qubits))
        self.theta_rz = nn.Parameter(torch.zeros(circuit_layers, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(circuit_layers, n_qubits)) if reupload else None
        nn.init.normal_(self.theta_ry, std=0.05)
        nn.init.normal_(self.theta_rz, std=0.05)
        self.decoder = nn.Linear(n_qubits, d_model)

    @property
    def circuit_parameter_count(self) -> int:
        total = self.theta_ry.numel() + self.theta_rz.numel()
        if self.input_scale is not None:
            total += self.input_scale.numel()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        angles = math.pi * torch.tanh(self.encoder(flat))
        batch = flat.shape[0]
        state = torch.zeros(batch, 2**self.n_qubits, dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j

        # Initial encoding. Re-upload variants inject the same data again in every layer.
        for q in range(self.n_qubits):
            state = _apply_single_qubit_gate(state, _ry(angles[:, q]), q, self.n_qubits)

        for layer in range(self.circuit_layers):
            if self.reupload:
                for q in range(self.n_qubits):
                    state = _apply_single_qubit_gate(
                        state,
                        _ry(angles[:, q] * self.input_scale[layer, q]),
                        q,
                        self.n_qubits,
                    )
            for q in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(self.theta_ry[layer, q].expand(batch)), q, self.n_qubits)
                state = _apply_single_qubit_gate(state, _rz(self.theta_rz[layer, q].expand(batch)), q, self.n_qubits)
            for control, target in _entangle_pairs(self.n_qubits, self.topology, layer):
                perm = _cnot_permutation(self.n_qubits, control, target, x.device)
                state = state[:, perm]

            # Training-only dissipation-inspired contraction. This is not a
            # faithful open-system channel; it is a controlled non-unitary
            # regularizer that biases the state mildly toward |0...0>.
            # Experiments anneal strength to zero before evaluation.
            if self.training and self.training_dissipation > 0.0:
                gamma = float(self.training_dissipation)
                anchor = torch.zeros_like(state)
                anchor[:, 0] = 1.0 + 0.0j
                state = (1.0 - gamma) * state + gamma * anchor
                state = state / state.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        probs = state.abs().square()
        basis = torch.arange(2**self.n_qubits, device=x.device)
        expectations = []
        for q in range(self.n_qubits):
            mask = 1 << (self.n_qubits - 1 - q)
            signs = torch.where((basis & mask) == 0, 1.0, -1.0)
            expectations.append((probs * signs).sum(dim=-1))
        z = torch.stack(expectations, dim=-1)
        return self.decoder(z).view(*original_shape)
