from __future__ import annotations

import torch


DEFAULT_TEXT = (
    "quantum circuits can act as tiny nonlinear bottlenecks. "
    "transformers predict the next token from previous tokens. "
    "we compare a classical feed forward network with a simulated quantum circuit. "
    "the goal is not quantum advantage but a controlled compression experiment. "
) * 200


class CharDataset:
    def __init__(self, text: str = DEFAULT_TEXT, block_size: int = 64, split: float = 0.9):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.block_size = block_size
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        cut = int(len(data) * split)
        self.train = data[:cut]
        self.val = data[cut:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def batch(self, split: str, batch_size: int, device: torch.device):
        data = self.train if split == "train" else self.val
        starts = torch.randint(0, len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts]).to(device)
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        fallback = self.stoi.get(" ", 0)
        return torch.tensor([[self.stoi.get(c, fallback) for c in text]], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in ids.flatten().tolist())
