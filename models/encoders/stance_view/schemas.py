from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StanceReplyExampleMeta:
    event_id: str
    dataset_id: str
    post_id: str
    parent_post_id: Optional[str]
    source_post_id: str
    stable_order: int
    depth: Optional[int]
    interaction_type: str
    kept_reason: str
    sampling_position: str
    reply_label_name: Optional[str] = None
    reply_label_id: int = -1
    reply_label_mask: int = 0
    context_cat: list[int] = field(default_factory=list)
    context_dense: list[float] = field(default_factory=list)


@dataclass
class StanceEventMeta:
    event_index: int
    event_id: str
    dataset_id: str
    platform: str
    canonical_split: str
    raw_split_source: Optional[str]
    source_post_id: str
    source_text: str
    original_usable_reply_count: int
    sampled_reply_count: int
    truncated: bool
    row_start_index: int
    row_end_index: int
    shard_id: int | None = None
    row_start_in_shard: int | None = None
    row_end_in_shard: int | None = None
    primary_task_name: Optional[str] = None
    primary_metric_name: Optional[str] = None
    primary_label_name: Optional[str] = None
    primary_label_id: int = -1
    observed_view_mask: dict[str, int] | None = None
    view_availability: dict[str, dict] | None = None


@dataclass
class StanceArrayShard:
    shard_id: int
    reply_embedding: np.ndarray
    context_cat: np.ndarray
    context_dense: np.ndarray
    reply_label_ids: np.ndarray
    reply_label_mask: np.ndarray
    event_index: np.ndarray
    event_sampled_reply_count: np.ndarray
    event_truncation_flag: np.ndarray


@dataclass
class StanceEncoderOutput:
    reply_hidden: "torch.Tensor"
    event_hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
    reply_logits: "torch.Tensor | None" = None
