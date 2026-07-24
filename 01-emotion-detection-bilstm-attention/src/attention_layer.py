"""PyTorch temporal attention layer."""
from __future__ import annotations
import torch
from torch import nn

class TemporalAttention(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_size, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        logits = torch.tanh(self.score(sequence).squeeze(-1))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights
