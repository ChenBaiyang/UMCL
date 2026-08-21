from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.evidence_view.config import EvidenceViewConfig, Mode


class SummaryEncoder(nn.Module):
    def __init__(self, config: EvidenceViewConfig, mode: Mode, input_dim: int, quality_dim: int) -> None:
        super().__init__()
        hidden_dim = config.summary_hidden_dim(mode)
        self.net = nn.Sequential(
            nn.Linear(input_dim + quality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.summary_branch_hidden_dim_value(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.summary_branch_hidden_dim_value(mode)),
        )

    def forward(self, summary_features: torch.Tensor, summary_quality_flags: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([summary_features, summary_quality_flags], dim=-1))


class ReferenceFeatureEncoder(nn.Module):
    def __init__(self, config: EvidenceViewConfig, mode: Mode) -> None:
        super().__init__()
        emb_dim = config.ref_embedding_dim(mode)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(config.ref_type_vocab), emb_dim),
                nn.Embedding(len(config.authority_bucket_vocab), emb_dim),
                nn.Embedding(2, emb_dim),
                nn.Embedding(2, emb_dim),
                nn.Embedding(2, emb_dim),
                nn.Embedding(2, emb_dim),
                nn.Embedding(2, emb_dim),
                nn.Embedding(len(config.source_channel_vocab), emb_dim),
            ]
        )
        self.net = nn.Sequential(
            nn.Linear(len(self.embeddings) * emb_dim, config.ref_hidden_dim(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ref_hidden_dim(mode), config.reference_branch_hidden_dim_value(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.reference_branch_hidden_dim_value(mode)),
        )

    def forward(self, ref_row_features: torch.Tensor) -> torch.Tensor:
        if ref_row_features.numel() == 0:
            return ref_row_features.new_zeros((*ref_row_features.shape[:-1], 0))
        categorical = ref_row_features.long()
        embedded = [emb(categorical[..., idx]) for idx, emb in enumerate(self.embeddings)]
        return self.net(torch.cat(embedded, dim=-1))
