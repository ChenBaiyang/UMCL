from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.evidence_view.config import EvidenceViewConfig, Mode
from models.encoders.evidence_view.models.context import ReferenceFeatureEncoder, SummaryEncoder
from models.encoders.evidence_view.models.heads import PosteriorHeads
from models.encoders.evidence_view.models.pooling import AttentionPooling, masked_mean_pool, mean_pool
from models.encoders.evidence_view.schemas import EvidenceEncoderOutput


class EvidenceEncoder(nn.Module):
    def __init__(self, config: EvidenceViewConfig, mode: Mode, summary_feature_dim: int, quality_feature_dim: int, ref_feature_dim: int, *, latent_dim: int) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        summary_branch_hidden_dim = config.summary_branch_hidden_dim_value(mode)
        reference_branch_hidden_dim = config.reference_branch_hidden_dim_value(mode)
        event_hidden_dim = config.event_hidden_dim_value(mode)
        self.summary_encoder = SummaryEncoder(config, mode, summary_feature_dim, quality_feature_dim)
        self.reference_encoder = ReferenceFeatureEncoder(config, mode) if ref_feature_dim > 0 else None
        self.attention_pool = AttentionPooling(reference_branch_hidden_dim, config.reference_attention_pooling_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(summary_branch_hidden_dim + reference_branch_hidden_dim, config.fusion_dim(mode)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim(mode), event_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(event_hidden_dim),
        )
        self.posterior = PosteriorHeads(event_hidden_dim, latent_dim)

    def _pool_refs(self, encoded_refs: torch.Tensor, ref_row_mask: torch.Tensor) -> torch.Tensor:
        if encoded_refs.numel() == 0:
            return ref_row_mask.new_zeros((ref_row_mask.shape[0], self.config.reference_branch_hidden_dim_value(self.mode)), dtype=torch.float32)
        if self.config.set_pooling == "mean":
            return mean_pool(encoded_refs, ref_row_mask)
        if self.config.set_pooling == "attention":
            return self.attention_pool(encoded_refs, ref_row_mask)
        return masked_mean_pool(encoded_refs, ref_row_mask)

    def forward(
        self,
        summary_features: torch.Tensor,
        summary_quality_flags: torch.Tensor,
        ref_row_features: torch.Tensor,
        ref_row_mask: torch.Tensor,
    ) -> EvidenceEncoderOutput:
        summary_hidden = self.summary_encoder(summary_features, summary_quality_flags)
        if self.reference_encoder is None or ref_row_features.numel() == 0 or ref_row_features.shape[1] == 0:
            reference_hidden = summary_features.new_zeros((summary_features.shape[0], self.config.reference_branch_hidden_dim_value(self.mode)))
        else:
            encoded_refs = self.reference_encoder(ref_row_features)
            reference_hidden = self._pool_refs(encoded_refs, ref_row_mask)
        event_hidden = self.fusion(torch.cat([reference_hidden, summary_hidden], dim=-1))
        mu, logvar = self.posterior(event_hidden)
        return EvidenceEncoderOutput(summary_hidden=summary_hidden, reference_hidden=reference_hidden, event_hidden=event_hidden, mu=mu, logvar=logvar)
