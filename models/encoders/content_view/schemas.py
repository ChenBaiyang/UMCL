from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ContentIndexRecord:
    event_id: str
    dataset_id: str
    platform: str
    canonical_split: str
    raw_split_source: Optional[str]
    primary_task_name: Optional[str] = None
    primary_metric_name: Optional[str] = None
    primary_label_name: Optional[str] = None
    primary_label_id: Optional[int] = None
    encoder_text_length_chars: int = 0
    encoder_text_length_tokens_approx: int = 0
    is_text_missing: bool = False
    is_text_empty: bool = False
    is_text_boilerplate: bool = False
    is_text_substantive: bool = False
    is_semantically_usable: bool = False
    has_url: int = 0
    has_media: int = 0
    surface_feature_dim: int = 0
    shard_id: int = 0
    row_in_shard: int = 0


@dataclass
class ContentArrayShard:
    shard_id: int
    event_ids: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    surface_features: np.ndarray | None = None
    quality_flags: np.ndarray | None = None
    primary_label_ids: np.ndarray | None = None


@dataclass
class ContentEncoderOutput:
    pooled: "torch.Tensor"
    hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
