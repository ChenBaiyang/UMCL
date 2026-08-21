from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SourceEventMeta:
    event_id: str
    dataset_id: str
    platform: str
    canonical_split: str
    raw_split_source: Optional[str]
    primary_task_name: Optional[str]
    primary_metric_name: Optional[str]
    primary_label_name: Optional[str]
    primary_label_id: int
    source_user_id: Optional[str]
    source_mode: str
    source_signal_observed: bool
    source_sparse_mask: bool
    source_identity_sparse_mask: bool
    source_social_sparse_mask: bool
    source_role_sparse_mask: bool
    source_history_sparse_mask: bool
    verified_flag: float
    has_account_created_at: float
    account_age_days: float
    has_profile_description: float
    has_display_name: float
    profile_description_length_chars: float
    display_name_length_chars: float
    profile_completeness_count: float
    avatar_present_flag: float
    location_present_flag: float
    gender_present_flag: float
    verified_reason_present_flag: float
    verified_type_present_flag: float
    followers_count_raw: float
    following_count_raw: float
    follower_followee_ratio: float
    post_count_raw: float
    listed_count_raw: float
    membership_indicator_bi_followers_raw: float
    favourites_count_raw: float
    geo_enabled_flag: float
    lifetime_posting_rate_proxy: float
    explicit_role_present: float
    verified_reason_present: float
    verified_type_id: int
    role_bucket_id: int
    age_bucket_id: int
    activity_rate_bucket_id: int
    summary_feature_dim: int
    notes: str = ""


@dataclass
class SourceArrayShard:
    shard_id: int
    event_ids: np.ndarray
    summary_features: np.ndarray
    summary_quality_flags: np.ndarray
    categorical_features: np.ndarray
    primary_label_ids: np.ndarray
    n_events: int


@dataclass
class SourceBatch:
    event_ids: list[str]
    summary_features: "torch.Tensor"
    summary_quality_flags: "torch.Tensor"
    categorical_features: "torch.Tensor"
    primary_label_ids: "torch.Tensor"


@dataclass
class SourceEncoderOutput:
    summary_hidden: "torch.Tensor"
    categorical_hidden: "torch.Tensor"
    event_hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
