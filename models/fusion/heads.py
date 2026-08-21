from __future__ import annotations

import torch
import torch.nn as nn


class GaussianReconstructionHead(nn.Module):
    def __init__(self, posterior_latent_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(posterior_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.logvar_parameter = nn.Parameter(torch.zeros(output_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.decoder(z)
        logvar = self.logvar_parameter.unsqueeze(0).expand_as(mean)
        return mean, logvar


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # 关键修复1：重新初始化最后一层，使用更小的初始化方差
        # 这确保初始logits的scale合理，避免softmax梯度消失
        nn.init.xavier_uniform_(self.layers[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layers[3].weight, gain=1.0)
        nn.init.zeros_(self.layers[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
