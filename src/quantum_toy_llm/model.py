from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .quantum import QuantumBottleneck
from .tensor_train import TensorTrainFFN


@dataclass
class TinyGPTConfig:
    vocab_size: int
    block_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    ffn_mult: int = 4
    mode: str = "classical"  # classical | bottleneck | quantum
    n_qubits: int = 4
    circuit_layers: int = 2
    quantum_reupload: bool = False
    quantum_topology: str = "ring"
    tt_rank: int = 2


class ClassicalFFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottleneckFFN(nn.Module):
    """Classical control with the same narrow latent width as the quantum interface."""
    def __init__(self, d_model: int, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, width),
            nn.GELU(),
            nn.Linear(width, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumFFN(nn.Module):
    def __init__(self, d_model: int, n_qubits: int, circuit_layers: int, dropout: float, reupload: bool = False, topology: str = "ring"):
        super().__init__()
        self.quantum = QuantumBottleneck(d_model, n_qubits, circuit_layers, reupload=reupload, topology=topology)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.quantum(x)))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.mode == "classical":
            self.ffn = ClassicalFFN(cfg.d_model, cfg.ffn_mult, cfg.dropout)
        elif cfg.mode == "bottleneck":
            self.ffn = BottleneckFFN(cfg.d_model, cfg.n_qubits, cfg.dropout)
        elif cfg.mode == "quantum":
            self.ffn = QuantumFFN(cfg.d_model, cfg.n_qubits, cfg.circuit_layers, cfg.dropout, cfg.quantum_reupload, cfg.quantum_topology)
        elif cfg.mode == "mps":
            self.ffn = TensorTrainFFN(cfg.d_model, rank=cfg.tt_rank, dropout=cfg.dropout)
        else:
            raise ValueError(f"unknown mode: {cfg.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError("sequence longer than block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 80, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
