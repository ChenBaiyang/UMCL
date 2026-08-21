from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.source_view.config import Mode, SourceViewConfig
from models.encoders.source_view.models.context import CategoricalEncoder, SummaryEncoder
from models.encoders.source_view.models.heads import PosteriorHeads
from models.encoders.source_view.schemas import SourceEncoderOutput


class SourceEncoder(nn.Module):
    def __init__(self, config: SourceViewConfig, mode: Mode, summary_feature_dim: int, quality_feature_dim: int, categorical_feature_dim: int, *, latent_dim: int) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        summary_branch_hidden_dim = config.summary_branch_hidden_dim_value(mode)
        categorical_branch_hidden_dim = config.categorical_branch_hidden_dim_value(mode)
        event_hidden_dim = config.event_hidden_dim_value(mode)
        self.summary_encoder = SummaryEncoder(config, mode, summary_feature_dim, quality_feature_dim)
        self.categorical_encoder = CategoricalEncoder(config, mode) if categorical_feature_dim > 0 else None
        self.fusion = nn.Sequential(
            nn.Linear(summary_branch_hidden_dim + categorical_branch_hidden_dim, config.fusion_dim(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim(mode), event_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(event_hidden_dim),
        )
        self.posterior = PosteriorHeads(event_hidden_dim, latent_dim)

    def forward(self, summary_features: torch.Tensor, summary_quality_flags: torch.Tensor, categorical_features: torch.Tensor) -> SourceEncoderOutput:
        summary_hidden = self.summary_encoder(summary_features, summary_quality_flags)
        if self.categorical_encoder is None or categorical_features.numel() == 0:
            categorical_hidden = summary_features.new_zeros((summary_features.shape[0], self.config.categorical_branch_hidden_dim_value(self.mode)))
        else:
            categorical_hidden = self.categorical_encoder(categorical_features)
        event_hidden = self.fusion(torch.cat([summary_hidden, categorical_hidden], dim=-1))
        mu, logvar = self.posterior(event_hidden)
        return SourceEncoderOutput(
            summary_hidden=summary_hidden,
            event_hidden=event_hidden,
            categorical_hidden=categorical_hidden,
            mu=mu,
            logvar=logvar,
        )
