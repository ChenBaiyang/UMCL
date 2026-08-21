from __future__ import annotations

import torch.nn as nn


class PosteriorHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, hidden):
        return self.mu(hidden), self.logvar(hidden)


class ReplyClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, hidden):
        return self.classifier(hidden)
