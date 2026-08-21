from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PropagationNodeExampleMeta:
    event_id: str
    dataset_id: str
    platform: str
    node_post_id: str
    parent_post_id: Optional[str]
    parent_index: int
    is_source: bool
    node_order_index: int
    time_observation_kind: str
    time_observation_kind_id: int
    created_at: Optional[str]
    time_from_source_sec: Optional[float]
    time_from_source_min: Optional[float]
    time_bucket_id: int
    stage_bucket_id: int
    depth_value: Optional[int]
    depth_bucket_id: int
    root_distance: Optional[int]
    parent_is_source: bool
    parent_inferred: bool
    interaction_type: str
    interaction_type_id: int
    text_status: str
    text_status_id: int
    is_placeholder: bool
    is_observed_text: bool
    is_boilerplate_text: bool
    has_continuous_time: bool
    log1p_child_count: float = 0.0
    is_leaf: bool = False
    order_bucket_id: int = 0
    notes: str = ""


@dataclass
class PropagationEventMeta:
    event_id: str
    dataset_id: str
    platform: str
    canonical_split: str
    raw_split_source: Optional[str]
    primary_task_name: Optional[str]
    primary_metric_name: Optional[str]
    primary_label_name: Optional[str]
    primary_label_id: int
    propagation_mode: str
    raw_n_nodes: int
    n_candidate_nodes: int
    n_encoded_nodes: int
    node_count_value: int
    node_count_bucket_id: int
    n_observed_timestamps: int
    n_relative_delay_only_nodes: int
    n_missing_time_nodes: int
    node_start_index: int
    node_end_index: int
    shard_id: int | None = None
    node_start_in_shard: int | None = None
    node_end_in_shard: int | None = None
    notes: str = ""


@dataclass
class PropagationArrayShard:
    shard_id: int
    event_ids: np.ndarray
    node_post_ids: np.ndarray
    node_features_continuous: np.ndarray
    node_features_categorical: np.ndarray
    node_masks: np.ndarray
    parent_index: np.ndarray
    node_sampling_role_ids: np.ndarray
    event_index: np.ndarray
    label_ids: np.ndarray
    label_mask: np.ndarray
    event_summary_features: np.ndarray
    temporal_profile_features: np.ndarray
    max_nodes_per_event: int
    n_node_examples: int


@dataclass
class PropagationBatch:
    event_ids: list[str]
    node_post_ids: list[str]
    node_features_continuous: "torch.Tensor"
    node_features_categorical: "torch.Tensor"
    node_masks: "torch.Tensor"
    parent_index: "torch.Tensor"
    event_index: "torch.Tensor"
    label_ids: "torch.Tensor"
    label_mask: "torch.Tensor"
    event_summary_features: "torch.Tensor"
    temporal_profile_features: "torch.Tensor"


@dataclass
class PropagationEncoderOutput:
    node_hidden: "torch.Tensor"
    event_hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
    node_aux_logits: "torch.Tensor | None" = None
