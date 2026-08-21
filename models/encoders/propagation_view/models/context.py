from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.propagation_view.config import PropagationViewConfig


class PropagationNodeFeatureEncoder(nn.Module):
    def __init__(self, config: PropagationViewConfig, categorical_feature_dim: int, continuous_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(32, config.categorical_feature_embedding_dim) for _ in range(categorical_feature_dim)])
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_feature_dim, config.continuous_feature_hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
        )
        self.structure_proj = nn.Sequential(
            nn.Linear(categorical_feature_dim * config.categorical_feature_embedding_dim + config.continuous_feature_hidden_dim, config.structure_feature_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(categorical_feature_dim * config.categorical_feature_embedding_dim + config.continuous_feature_hidden_dim + config.structure_feature_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, categorical_features: torch.Tensor, continuous_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = [emb(categorical_features[:, idx]) for idx, emb in enumerate(self.embeddings)]
        categorical_hidden = torch.cat(embedded, dim=-1)
        continuous_hidden = self.continuous_mlp(continuous_features)
        structure_hidden = self.structure_proj(torch.cat([categorical_hidden, continuous_hidden], dim=-1))
        node_input_hidden = self.node_proj(torch.cat([categorical_hidden, continuous_hidden, structure_hidden], dim=-1))
        return node_input_hidden, structure_hidden
