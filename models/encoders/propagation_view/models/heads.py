from __future__ import annotations

import torch
import torch.nn as nn


class PosteriorHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(hidden), self.logvar(hidden)


class NodeAuxHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)
