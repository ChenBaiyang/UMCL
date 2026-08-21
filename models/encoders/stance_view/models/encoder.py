from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.stance_view.config import Mode, StanceViewConfig
from models.encoders.stance_view.models.heads import PosteriorHeads, ReplyClassifierHead
from models.encoders.stance_view.models.reply_encoder import StanceReplyEncoder
from models.encoders.stance_view.models.summarizer import BucketedSetSummarizer
from models.encoders.stance_view.schemas import StanceEncoderOutput


class StanceEncoder(nn.Module):
    def __init__(self, config: StanceViewConfig, mode: Mode, enable_reply_logits: bool = True, *, latent_dim: int) -> None:
        super().__init__()
        self.reply_encoder = StanceReplyEncoder(config, mode)
        self.reply_classifier = ReplyClassifierHead(config.reply_hidden_dim(mode), len(config.stance_label_names)) if enable_reply_logits else None
        self.summarizer = BucketedSetSummarizer(config, mode)
        self.posterior_heads = PosteriorHeads(config.event_hidden_dim, latent_dim)

    def forward(
        self,
        reply_embedding: torch.Tensor,
        context_cat: torch.Tensor,
        context_dense: torch.Tensor,
        event_index: torch.Tensor,
        event_sampled_reply_count: torch.Tensor,
        event_truncation_flag: torch.Tensor,
    ) -> StanceEncoderOutput:
        reply_hidden = self.reply_encoder(reply_embedding, context_cat, context_dense)
        reply_logits = self.reply_classifier(reply_hidden) if self.reply_classifier is not None else None
        if reply_logits is None:
            reply_probs = torch.zeros((reply_hidden.shape[0], 4), device=reply_hidden.device, dtype=reply_hidden.dtype)
        else:
            reply_probs = torch.softmax(reply_logits, dim=-1)
        event_hidden = self.summarizer(
            reply_hidden=reply_hidden,
            reply_probs=reply_probs,
            context_cat=context_cat,
            context_dense=context_dense,
            event_index=event_index,
            event_sampled_reply_count=event_sampled_reply_count,
            event_truncation_flag=event_truncation_flag,
        )
        mu, logvar = self.posterior_heads(event_hidden)
        return StanceEncoderOutput(
            reply_hidden=reply_hidden,
            event_hidden=event_hidden,
            mu=mu,
            logvar=logvar,
            reply_logits=reply_logits,
        )
