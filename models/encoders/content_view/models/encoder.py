from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.content_view.config import ContentViewConfig, Mode
from models.encoders.content_view.models.backbones import load_backbone
from models.encoders.content_view.models.heads import PosteriorHeads, SurfaceFeatureMLP
from models.encoders.content_view.models.pooling import pool_sentence
from models.encoders.content_view.schemas import ContentEncoderOutput


class ContentHiddenEncoder(nn.Module):
    def __init__(self, config: ContentViewConfig, mode: Mode, surface_feature_dim: int = 0) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.enable_surface_fusion = bool(config.enable_surface_fusion and surface_feature_dim > 0)

        self.backbone, backbone_hidden_dim = load_backbone(config, mode)
        self.text_projection = nn.Sequential(
            nn.Linear(backbone_hidden_dim, config.event_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        if self.enable_surface_fusion:
            self.surface_mlp = SurfaceFeatureMLP(surface_feature_dim, config.surface_feature_hidden_dim, config.dropout)
            fusion_input_dim = config.event_hidden_dim + config.surface_feature_hidden_dim
        else:
            self.surface_mlp = None
            fusion_input_dim = config.event_hidden_dim

        self.hidden_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, config.event_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.event_hidden_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        surface_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_sentence(outputs.last_hidden_state, attention_mask, self.config.content_sentence_pool_mode)
        text_hidden = self.text_projection(pooled)
        if self.enable_surface_fusion and surface_features is not None:
            surface_hidden = self.surface_mlp(surface_features)
            fused = torch.cat([text_hidden, surface_hidden], dim=-1)
        else:
            fused = text_hidden
        hidden = self.hidden_projection(fused)
        return pooled, hidden


class ContentEncoder(nn.Module):
    def __init__(self, config: ContentViewConfig, mode: Mode, surface_feature_dim: int = 0, *, latent_dim: int) -> None:
        super().__init__()
        self.hidden_encoder = ContentHiddenEncoder(
            config,
            mode,
            surface_feature_dim=surface_feature_dim,
        )
        self.posterior_heads = PosteriorHeads(config.event_hidden_dim, latent_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        surface_features: torch.Tensor | None = None,
    ) -> ContentEncoderOutput:
        pooled, hidden = self.hidden_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            surface_features=surface_features,
        )
        mu, logvar = self.posterior_heads(hidden)
        return ContentEncoderOutput(pooled=pooled, hidden=hidden, mu=mu, logvar=logvar)
