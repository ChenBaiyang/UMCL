from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.propagation_view.config import Mode, PropagationViewConfig
from models.encoders.propagation_view.models.aggregator import SequenceTransformer
from models.encoders.propagation_view.models.backbones import propagation_model_dims
from models.encoders.propagation_view.models.context import PropagationNodeFeatureEncoder
from models.encoders.propagation_view.models.heads import NodeAuxHead, PosteriorHeads
from models.encoders.propagation_view.schemas import PropagationEncoderOutput


class PropagationEncoder(nn.Module):
    def __init__(
        self,
        config: PropagationViewConfig,
        mode: Mode,
        categorical_feature_dim: int,
        continuous_feature_dim: int,
        event_summary_feature_dim: int = 0,
        temporal_profile_feature_dim: int = 0,
        enable_node_aux_head: bool | None = None,
        *,
        latent_dim: int,
    ) -> None:
        super().__init__()
        dims = propagation_model_dims(config, mode)
        hidden_dim = int(dims["hidden_dim"])
        dropout = float(dims["dropout"])
        self.stage_bucket_index = 13 if categorical_feature_dim > 13 else None
        self.node_encoder = PropagationNodeFeatureEncoder(
            config=config,
            categorical_feature_dim=categorical_feature_dim,
            continuous_feature_dim=continuous_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.transformer = SequenceTransformer(
            hidden_dim=hidden_dim,
            num_heads=int(dims["heads"]),
            num_layers=int(dims["layers"]),
            ffn_hidden_dim=int(dims["ffn_hidden_dim"]),
            dropout=dropout,
        )
        summary_input_dim = int(event_summary_feature_dim) + int(temporal_profile_feature_dim)
        summary_hidden_dim = hidden_dim
        self.summary_encoder = None
        if summary_input_dim > 0:
            self.summary_encoder = nn.Sequential(
                nn.Linear(summary_input_dim, summary_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(summary_hidden_dim),
            )
        pooled_hidden_factor = 5
        fusion_input_dim = hidden_dim * pooled_hidden_factor + (summary_hidden_dim if self.summary_encoder is not None else 0)
        self.event_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, config.event_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(config.event_hidden_dim),
        )
        self.posterior = PosteriorHeads(config.event_hidden_dim, latent_dim)
        enable_aux = config.aux_node_head_enabled if enable_node_aux_head is None else enable_node_aux_head
        self.node_aux_head = NodeAuxHead(hidden_dim, 4) if enable_aux else None
        self.config = config
        self.event_hidden_dim = config.event_hidden_dim

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = hidden.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        values = masked.max(dim=1).values
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))

    def _stage_pool(self, hidden: torch.Tensor, valid_mask: torch.Tensor, padded_categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.stage_bucket_index is None:
            zero = hidden.new_zeros((hidden.shape[0], hidden.shape[-1]))
            return zero, zero, zero
        stage_bucket = padded_categorical[:, :, self.stage_bucket_index]
        early_mask = valid_mask & (stage_bucket == 1)
        mid_mask = valid_mask & (stage_bucket == 2)
        late_mask = valid_mask & (stage_bucket == 3)
        return (
            self._masked_mean(hidden, early_mask),
            self._masked_mean(hidden, mid_mask),
            self._masked_mean(hidden, late_mask),
        )

    def forward(
        self,
        node_features_continuous: torch.Tensor,
        node_features_categorical: torch.Tensor,
        node_masks: torch.Tensor,
        parent_index: torch.Tensor | None,
        event_index: torch.Tensor,
        event_summary_features: torch.Tensor | None = None,
        temporal_profile_features: torch.Tensor | None = None,
    ) -> PropagationEncoderOutput:
        node_input_hidden, _ = self.node_encoder(node_features_categorical, node_features_continuous)
        if node_input_hidden.numel() == 0:
            empty_event = node_input_hidden.new_zeros((0, self.event_hidden_dim))
            mu, logvar = self.posterior(empty_event)
            return PropagationEncoderOutput(node_hidden=node_input_hidden, event_hidden=empty_event, mu=mu, logvar=logvar, node_aux_logits=None)

        unique_events, counts = torch.unique_consecutive(event_index, return_counts=True)
        batch_size = int(unique_events.shape[0])
        max_len = int(counts.max().item())
        hidden_dim = node_input_hidden.shape[-1]
        total_nodes = int(node_input_hidden.shape[0])
        padded = node_input_hidden.new_zeros((batch_size, max_len, hidden_dim))
        padded_categorical = node_features_categorical.new_zeros((batch_size, max_len, node_features_categorical.shape[-1]))
        valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=node_input_hidden.device)
        padded_parent_index = torch.full((batch_size, max_len), -1, dtype=torch.long, device=node_input_hidden.device)

        event_ids = torch.arange(batch_size, device=node_input_hidden.device, dtype=torch.long)
        segment_ids = torch.repeat_interleave(event_ids, counts)
        segment_starts = torch.cumsum(counts, dim=0) - counts
        local_positions = torch.arange(total_nodes, device=node_input_hidden.device, dtype=torch.long) - torch.repeat_interleave(segment_starts, counts)
        flat_positions = segment_ids * max_len + local_positions

        padded.view(-1, hidden_dim)[flat_positions] = node_input_hidden
        padded_categorical.view(-1, node_features_categorical.shape[-1])[flat_positions] = node_features_categorical
        valid_mask.view(-1)[flat_positions] = node_masks > 0
        if parent_index is not None:
            padded_parent_index.view(-1)[flat_positions] = parent_index

        encoded = self.transformer(
            padded,
            valid_mask,
            padded_parent_index,
            parent_bias_value=float(self.config.parent_bias_value),
            source_bias_value=float(self.config.source_bias_value),
            parent_bias_attention_enabled=bool(self.config.parent_bias_attention_enabled),
        )
        node_hidden = encoded[valid_mask]
        root_hidden = encoded[:, 0, :]
        mean_hidden = self._masked_mean(encoded, valid_mask)
        max_hidden = self._masked_max(encoded, valid_mask)
        early_hidden, mid_hidden, late_hidden = self._stage_pool(encoded, valid_mask, padded_categorical)
        # pooled_hidden = torch.cat([root_hidden, mean_hidden, max_hidden, early_hidden, mid_hidden, late_hidden], dim=-1)
        pooled_hidden = torch.cat([root_hidden, max_hidden, early_hidden, mid_hidden, late_hidden], dim=-1)

        fusion_parts = [pooled_hidden]
        if self.summary_encoder is not None:
            summary_parts: list[torch.Tensor] = []
            if event_summary_features is not None and event_summary_features.numel() > 0:
                summary_parts.append(event_summary_features)
            if temporal_profile_features is not None and temporal_profile_features.numel() > 0:
                summary_parts.append(temporal_profile_features)
            summary_input = torch.cat(summary_parts, dim=-1) if summary_parts else pooled_hidden.new_zeros((batch_size, 0))
            if summary_input.shape[-1] > 0:
                fusion_parts.append(self.summary_encoder(summary_input))
        event_hidden = self.event_proj(torch.cat(fusion_parts, dim=-1))
        mu, logvar = self.posterior(event_hidden)
        node_aux_logits = self.node_aux_head(node_hidden) if self.node_aux_head is not None else None
        return PropagationEncoderOutput(
            node_hidden=node_hidden,
            event_hidden=event_hidden,
            mu=mu,
            logvar=logvar,
            node_aux_logits=node_aux_logits,
        )
