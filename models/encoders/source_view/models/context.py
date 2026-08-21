from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.source_view.config import Mode, SourceViewConfig


class SummaryEncoder(nn.Module):
    def __init__(self, config: SourceViewConfig, mode: Mode, summary_feature_dim: int, quality_feature_dim: int) -> None:
        super().__init__()
        hidden = config.summary_hidden_dim(mode)
        self.use_linear_shortcut = bool(config.summary_linear_shortcut)
        self.shortcut_proj = nn.Sequential(
            nn.Linear(summary_feature_dim, config.summary_branch_hidden_dim_value(mode)),
            nn.LayerNorm(config.summary_branch_hidden_dim_value(mode)),
        )
        self.net = nn.Sequential(
            nn.Linear(summary_feature_dim + quality_feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.summary_branch_hidden_dim_value(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.summary_branch_hidden_dim_value(mode)),
        )

    def forward(self, summary_features: torch.Tensor, summary_quality_flags: torch.Tensor) -> torch.Tensor:
        summary_hidden = self.net(torch.cat([summary_features, summary_quality_flags], dim=-1))
        if not self.use_linear_shortcut:
            return summary_hidden
        shortcut_hidden = self.shortcut_proj(summary_features)
        return summary_hidden + shortcut_hidden


class CategoricalEncoder(nn.Module):
    def __init__(self, config: SourceViewConfig, mode: Mode) -> None:
        super().__init__()
        emb_dim = config.cat_emb_dim(mode)
        self.embeddings = nn.ModuleDict(
            {
                "verified_type_id": nn.Embedding(len(config.verified_type_vocab), emb_dim),
                "role_bucket_id": nn.Embedding(len(config.role_bucket_vocab), emb_dim),
                "age_bucket_id": nn.Embedding(len(config.age_bucket_vocab), emb_dim),
                "activity_rate_bucket_id": nn.Embedding(len(config.activity_rate_bucket_vocab), emb_dim),
            }
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * len(self.embeddings), config.cat_hidden_dim(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cat_hidden_dim(mode), config.categorical_branch_hidden_dim_value(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.categorical_branch_hidden_dim_value(mode)),
        )

    def forward(self, categorical_features: torch.Tensor) -> torch.Tensor:
        chunks = []
        for idx, name in enumerate(self.embeddings.keys()):
            chunks.append(self.embeddings[name](categorical_features[:, idx]))
        return self.proj(torch.cat(chunks, dim=-1))
