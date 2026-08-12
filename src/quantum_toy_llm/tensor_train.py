from __future__ import annotations

import math
import torch
from torch import nn


class TTLinear(nn.Module):
    """Small tensor-train matrix layer reconstructed on the fly.

    Intended as a quantum-inspired compression control, not an optimized TT kernel.
    Input/output dimensions must equal the product of ``modes``.
    """
    def __init__(self, features: int, rank: int = 2, modes=(4, 4, 2), bias: bool = True):
        super().__init__()
        if math.prod(modes) != features:
            raise ValueError("product(modes) must equal features")
        self.features = features
        self.modes = tuple(modes)
        self.rank = rank
        ranks = [1] + [rank] * (len(modes) - 1) + [1]
        self.cores = nn.ParameterList()
        for i, mode in enumerate(modes):
            core = nn.Parameter(torch.empty(ranks[i], mode, mode, ranks[i + 1]))
            nn.init.normal_(core, std=0.08 / math.sqrt(max(1, rank)))
            self.cores.append(core)
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None

    def dense_weight(self) -> torch.Tensor:
        # Contract TT ranks, leaving interleaved input/output mode axes.
        t = self.cores[0].squeeze(0)  # [in1, out1, r]
        for core in self.cores[1:]:
            t = torch.tensordot(t, core, dims=([-1], [0]))
        t = t.squeeze(-1)
        n = len(self.modes)
        perm = list(range(0, 2*n, 2)) + list(range(1, 2*n, 2))
        t = t.permute(*perm).contiguous()
        return t.view(self.features, self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dense_weight(), self.bias)


class TensorTrainFFN(nn.Module):
    def __init__(self, d_model: int, rank: int = 2, dropout: float = 0.0):
        super().__init__()
        self.up = TTLinear(d_model, rank=rank)
        self.down = TTLinear(d_model, rank=rank)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x))))
