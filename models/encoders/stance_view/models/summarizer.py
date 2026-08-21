from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.encoders.stance_view.config import Mode, StanceViewConfig


def _pack_by_event(
    reply_hidden: torch.Tensor,
    reply_probs: torch.Tensor,
    context_cat: torch.Tensor,
    context_dense: torch.Tensor,
    event_index: torch.Tensor,
    event_sampled_reply_count: torch.Tensor,
    event_truncation_flag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_events, counts = torch.unique_consecutive(event_index, return_counts=True)
    batch_size = int(unique_events.shape[0])
    max_len = int(counts.max().item())
    total_rows = int(reply_hidden.shape[0])

    event_ids = torch.arange(batch_size, device=reply_hidden.device, dtype=torch.long)
    segment_ids = torch.repeat_interleave(event_ids, counts)
    segment_starts = torch.cumsum(counts, dim=0) - counts
    local_positions = torch.arange(total_rows, device=reply_hidden.device, dtype=torch.long) - torch.repeat_interleave(segment_starts, counts)
    flat_positions = segment_ids * max_len + local_positions

    hidden_dim = reply_hidden.shape[-1]
    prob_dim = reply_probs.shape[-1]
    cat_dim = context_cat.shape[-1]
    dense_dim = context_dense.shape[-1]
    padded_hidden = reply_hidden.new_zeros((batch_size, max_len, hidden_dim))
    padded_probs = reply_probs.new_zeros((batch_size, max_len, prob_dim))
    padded_cat = context_cat.new_zeros((batch_size, max_len, cat_dim))
    padded_dense = context_dense.new_zeros((batch_size, max_len, dense_dim))
    padded_valid = torch.zeros((batch_size, max_len), dtype=torch.bool, device=reply_hidden.device)
    padded_sampled = event_sampled_reply_count.new_zeros((batch_size, max_len))
    padded_truncated = event_truncation_flag.new_zeros((batch_size, max_len))

    padded_hidden.view(-1, hidden_dim)[flat_positions] = reply_hidden
    padded_probs.view(-1, prob_dim)[flat_positions] = reply_probs
    padded_cat.view(-1, cat_dim)[flat_positions] = context_cat
    padded_dense.view(-1, dense_dim)[flat_positions] = context_dense
    padded_valid.view(-1)[flat_positions] = True
    padded_sampled.view(-1)[flat_positions] = event_sampled_reply_count
    padded_truncated.view(-1)[flat_positions] = event_truncation_flag
    return padded_hidden, padded_probs, padded_cat, padded_dense, padded_valid, padded_sampled, padded_truncated


class BucketedSetSummarizer(nn.Module):
    def __init__(self, config: StanceViewConfig, mode: Mode) -> None:
        super().__init__()
        reply_hidden_dim = config.reply_hidden_dim(mode)
        summary_hidden_dim = max(1, reply_hidden_dim // 4)
        bucket_count = 4
        prob_dim = len(config.stance_label_names)
        summary_dim = (summary_hidden_dim * (2 + bucket_count + bucket_count)) + 102
        hidden_dim = config.summarizer_hidden_dim(mode)
        self.summary_hidden_compressor = nn.Sequential(
            nn.Linear(reply_hidden_dim, summary_hidden_dim),
            nn.LayerNorm(summary_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.summary_norm = nn.LayerNorm(summary_dim)
        self.summary_projection = nn.Linear(summary_dim, hidden_dim)
        self.summary_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, config.event_hidden_dim)
        self.output_norm = nn.LayerNorm(config.event_hidden_dim)
        self.dropout_proj = nn.Dropout(config.dropout+0.1)
        self.activation = nn.GELU()
        self.max_replies_per_event = float(config.max_replies_per_event)
        self.bucket_values = (0, 1, 2, 3)
        self.prob_dim = prob_dim
        self.support_idx = config.stance_label_names.index("support")
        self.deny_idx = config.stance_label_names.index("deny")
        self.query_idx = config.stance_label_names.index("query")
        dense_index = {name: idx for idx, name in enumerate(config.context_dense_fields)}
        self.reply_source_cosine_idx = dense_index["reply_source_cosine"]
        self.reply_parent_cosine_idx = dense_index["reply_parent_cosine"]
        self.parent_source_cosine_idx = dense_index["parent_source_cosine"]
        self.source_overlap_ratio_idx = dense_index["source_overlap_ratio"]
        self.parent_overlap_ratio_idx = dense_index["parent_overlap_ratio"]
        self.has_parent_text_idx = dense_index["has_parent_text"]
        self.eps = 1e-8

    @staticmethod
    def _masked_ratio(mask: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = mask.sum(dim=1, keepdim=True).to(torch.float32)
        total = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float32)
        present = (count > 0).to(torch.float32)
        return count / total, present

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask.unsqueeze(-1).to(values.dtype)).sum(dim=1) / denom

    def _masked_scalar_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask.to(values.dtype)).sum(dim=1, keepdim=True) / denom

    def _topk_stats(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_keep = min(8, scores.shape[1])
        if max_keep <= 0:
            zeros = scores.new_zeros((scores.shape[0], 1))
            return zeros, zeros
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
        sorted_scores = torch.sort(masked_scores, dim=1, descending=True).values[:, :max_keep]
        reply_count = valid_mask.sum(dim=1, keepdim=True)
        k_event = reply_count.clamp_min(1).clamp_max(max_keep)
        positions = torch.arange(max_keep, device=scores.device).unsqueeze(0)
        keep_mask = positions < k_event
        safe_scores = torch.where(keep_mask, sorted_scores, torch.zeros_like(sorted_scores))
        max_score = sorted_scores[:, :1]
        max_score = torch.where(torch.isfinite(max_score), max_score, torch.zeros_like(max_score))
        topk_mean = safe_scores.sum(dim=1, keepdim=True) / k_event.to(scores.dtype)
        return max_score, topk_mean

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=-1, keepdim=True)

    def forward(
        self,
        reply_hidden: torch.Tensor,
        reply_probs: torch.Tensor,
        context_cat: torch.Tensor,
        context_dense: torch.Tensor,
        event_index: torch.Tensor,
        event_sampled_reply_count: torch.Tensor,
        event_truncation_flag: torch.Tensor,
    ) -> torch.Tensor:
        if reply_hidden.numel() == 0:
            return reply_hidden.new_zeros((0, self.output_projection.out_features))
        event_hidden, event_probs, event_cat, event_dense, valid_mask, event_sampled_reply_count, event_truncation_flag = _pack_by_event(
            reply_hidden,
            reply_probs,
            context_cat,
            context_dense,
            event_index,
            event_sampled_reply_count,
            event_truncation_flag,
        )
        valid_float = valid_mask.unsqueeze(-1).to(event_hidden.dtype)
        event_counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(event_hidden.dtype)
        compressed_event_hidden = self.summary_hidden_compressor(event_hidden)

        global_hidden_mean = (compressed_event_hidden * valid_float).sum(dim=1) / event_counts
        event_max = compressed_event_hidden.masked_fill(~valid_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        global_hidden_max = torch.where(torch.isfinite(event_max), event_max, torch.zeros_like(event_max))
        global_reaction_mean = (event_probs * valid_float).sum(dim=1) / event_counts

        time_hidden_parts: list[torch.Tensor] = []
        time_mask_parts: list[torch.Tensor] = []
        time_ratio_parts: list[torch.Tensor] = []
        time_reaction_parts: list[torch.Tensor] = []
        time_reaction_entropy_parts: list[torch.Tensor] = []
        time_reaction_map: dict[int, torch.Tensor] = {}
        for bucket_id in self.bucket_values:
            bucket_mask = valid_mask & (event_cat[:, :, 1] == bucket_id)
            ratio, present = self._masked_ratio(bucket_mask, valid_mask)
            hidden_mean = self._masked_mean(compressed_event_hidden, bucket_mask.to(compressed_event_hidden.dtype))
            reaction_mean = self._masked_mean(event_probs, bucket_mask.to(event_probs.dtype))
            time_hidden_parts.append(hidden_mean)
            time_mask_parts.append(present)
            time_ratio_parts.append(ratio)
            time_reaction_parts.append(reaction_mean)
            time_reaction_entropy_parts.append(self._entropy(reaction_mean))
            time_reaction_map[bucket_id] = reaction_mean

        depth_hidden_parts: list[torch.Tensor] = []
        depth_mask_parts: list[torch.Tensor] = []
        depth_ratio_parts: list[torch.Tensor] = []
        depth_reaction_parts: list[torch.Tensor] = []
        for bucket_id in self.bucket_values:
            bucket_mask = valid_mask & (event_cat[:, :, 0] == bucket_id)
            ratio, present = self._masked_ratio(bucket_mask, valid_mask)
            hidden_mean = self._masked_mean(compressed_event_hidden, bucket_mask.to(compressed_event_hidden.dtype))
            reaction_mean = self._masked_mean(event_probs, bucket_mask.to(event_probs.dtype))
            depth_hidden_parts.append(hidden_mean)
            depth_mask_parts.append(present)
            depth_ratio_parts.append(ratio)
            depth_reaction_parts.append(reaction_mean)

        direct_mask = valid_mask & (event_cat[:, :, 2] == 1)
        nondirect_mask = valid_mask & (event_cat[:, :, 2] == 0)
        direct_reaction_mean = self._masked_mean(event_probs, direct_mask.to(event_probs.dtype))
        nondirect_reaction_mean = self._masked_mean(event_probs, nondirect_mask.to(event_probs.dtype))
        _, direct_reaction_mask = self._masked_ratio(direct_mask, valid_mask)
        _, nondirect_reaction_mask = self._masked_ratio(nondirect_mask, valid_mask)

        early_reaction = time_reaction_map[1]
        late_reaction = time_reaction_map[3]
        early_skeptical = early_reaction[:, self.deny_idx : self.deny_idx + 1] + early_reaction[:, self.query_idx : self.query_idx + 1]
        late_skeptical = late_reaction[:, self.deny_idx : self.deny_idx + 1] + late_reaction[:, self.query_idx : self.query_idx + 1]
        skeptical_shift = [
            late_reaction[:, self.deny_idx : self.deny_idx + 1] - early_reaction[:, self.deny_idx : self.deny_idx + 1],
            late_reaction[:, self.query_idx : self.query_idx + 1] - early_reaction[:, self.query_idx : self.query_idx + 1],
            late_skeptical - early_skeptical,
            early_reaction[:, self.support_idx : self.support_idx + 1] - late_reaction[:, self.support_idx : self.support_idx + 1],
        ]

        deny_scores = event_probs[:, :, self.deny_idx]
        query_scores = event_probs[:, :, self.query_idx]
        skeptical_scores = deny_scores + query_scores
        max_deny, topk_mean_deny = self._topk_stats(deny_scores, valid_mask)
        max_query, topk_mean_query = self._topk_stats(query_scores, valid_mask)
        max_skeptical, topk_mean_skeptical = self._topk_stats(skeptical_scores, valid_mask)

        mean_reply_source_cosine = self._masked_scalar_mean(event_dense[:, :, self.reply_source_cosine_idx], valid_mask)
        mean_reply_parent_cosine = self._masked_scalar_mean(event_dense[:, :, self.reply_parent_cosine_idx], valid_mask)
        mean_parent_source_cosine = self._masked_scalar_mean(event_dense[:, :, self.parent_source_cosine_idx], valid_mask)
        mean_source_overlap_ratio = self._masked_scalar_mean(event_dense[:, :, self.source_overlap_ratio_idx], valid_mask)
        mean_parent_overlap_ratio = self._masked_scalar_mean(event_dense[:, :, self.parent_overlap_ratio_idx], valid_mask)
        has_parent_text_ratio = self._masked_scalar_mean(event_dense[:, :, self.has_parent_text_idx], valid_mask)
        parent_relation_gap = mean_reply_parent_cosine - mean_reply_source_cosine

        summary_parts = [
            global_hidden_mean,
            global_hidden_max,
            *time_hidden_parts,
            *time_mask_parts,
            *time_ratio_parts,
            *depth_hidden_parts,
            *depth_mask_parts,
            *depth_ratio_parts,
            global_reaction_mean,
            *time_reaction_parts,
            *depth_reaction_parts,
            direct_reaction_mean,
            nondirect_reaction_mean,
            *skeptical_shift,
            max_deny,
            max_query,
            max_skeptical,
            topk_mean_deny,
            topk_mean_query,
            topk_mean_skeptical,
            self._entropy(global_reaction_mean),
            *time_reaction_entropy_parts,
            mean_reply_source_cosine,
            mean_reply_parent_cosine,
            mean_parent_source_cosine,
            mean_source_overlap_ratio,
            mean_parent_overlap_ratio,
            has_parent_text_ratio,
            parent_relation_gap,
            *time_ratio_parts,
            *time_mask_parts,
            *depth_ratio_parts,
            *depth_mask_parts,
            direct_reaction_mask,
            nondirect_reaction_mask,
            torch.log1p(event_sampled_reply_count[:, :1].to(event_hidden.dtype)),
            event_truncation_flag[:, :1].to(event_hidden.dtype),
        ]
        summary_vector = torch.cat(summary_parts, dim=-1)
        hidden = self.summary_norm(summary_vector)
        hidden = self.summary_projection(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout_proj(hidden)
        gate = torch.sigmoid(self.summary_gate(hidden))
        hidden = hidden * gate
        event_hidden = self.output_projection(hidden)
        event_hidden = self.output_norm(event_hidden)
        return self.activation(event_hidden)
