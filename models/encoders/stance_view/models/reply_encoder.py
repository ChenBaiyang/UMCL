from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.stance_view.config import Mode, StanceViewConfig


class _MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StanceReplyEncoder(nn.Module):
    def __init__(self, config: StanceViewConfig, mode: Mode) -> None:
        super().__init__()
        artifact_mode = config.artifact_mode(mode)
        embedding_input_dim = config.reply_embedding_dim_value(mode)
        embedding_proj_dim = config.reply_embedding_proj_dim(mode)
        context_hidden_dim = config.context_hidden_dim(mode)
        dense_hidden_dim = config.dense_hidden_dim(mode)
        reply_hidden_dim = config.reply_hidden_dim(mode)
        fusion_hidden_dim = config.reply_mlp_hidden_dim(mode)

        self.depth_embedding = nn.Embedding(4, 8)
        self.time_embedding = nn.Embedding(4, 8)
        self.parent_embedding = nn.Embedding(2, 4)
        self.interaction_embedding = nn.Embedding(5, 8)
        context_input_dim = 8 + 8 + 4 + 8

        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_input_dim, embedding_proj_dim),
            nn.LayerNorm(embedding_proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(context_input_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.dense_projection = _MLPBlock(len(config.context_dense_fields), dense_hidden_dim, dense_hidden_dim, config.dropout)
        self.reply_fusion = nn.Sequential(
            nn.Linear(embedding_proj_dim + context_hidden_dim + dense_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(fusion_hidden_dim, reply_hidden_dim),
            nn.LayerNorm(reply_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.artifact_mode = artifact_mode

    def forward(self, reply_embedding: torch.Tensor, context_cat: torch.Tensor, context_dense: torch.Tensor) -> torch.Tensor:
        context_hidden = torch.cat(
            [
                self.depth_embedding(context_cat[:, 0].clamp_min(0)),
                self.time_embedding(context_cat[:, 1].clamp_min(0)),
                self.parent_embedding(context_cat[:, 2].clamp(0, 1)),
                self.interaction_embedding(context_cat[:, 3].clamp_min(0)),
            ],
            dim=-1,
        )
        embedding_hidden = self.embedding_projection(reply_embedding)
        context_hidden = self.context_projection(context_hidden)
        dense_hidden = self.dense_projection(context_dense)
        return self.reply_fusion(torch.cat([embedding_hidden, context_hidden, dense_hidden], dim=-1))
