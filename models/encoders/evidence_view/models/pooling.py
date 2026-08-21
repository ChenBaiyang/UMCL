from __future__ import annotations

import torch
import torch.nn as nn


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mean_pool(hidden, mask)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(hidden).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, float("-inf"))
        all_masked = mask.sum(dim=1) == 0
        if all_masked.any():
            scores = scores.clone()
            scores[all_masked] = 0.0
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)
