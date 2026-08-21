from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EvidenceEventMeta:
    event_id: str
    dataset_id: str
    platform: str
    canonical_split: str
    raw_split_source: Optional[str]
    primary_task_name: Optional[str]
    primary_metric_name: Optional[str]
    primary_label_name: Optional[str]
    primary_label_id: int
    source_post_id: str
    evidence_mode: str
    has_explicit_ref: bool
    has_any_source_evidence_signal: bool
    n_refs_total: int
    n_url_refs: int
    n_media_refs: int
    n_internal_refs: int
    n_external_refs: int
    n_reference_types_present: int
    has_multi_ref_type: bool
    has_multi_ref_rows: bool
    has_specific_reference_signal: bool
    evidence_signal_observed: bool
    no_reference_rows_mask: bool
    weak_evidence_mask: bool
    sparse_evidence_mask: bool
    authority_unknown_mask: bool
    summary_feature_dim: int
    notes: str = ""


@dataclass
class EvidenceReferenceRowMeta:
    event_id: str
    dataset_id: str
    platform: str
    ref_id: str
    ref_type: str
    domain: Optional[str]
    authority_bucket_name: str
    is_internal_platform_ref: Optional[bool]
    has_normalized_value: bool
    has_domain: bool
    has_target_post_id: bool
    source_channel: str
    ref_order: int
    notes: str = ""


@dataclass
class EvidenceArrayShard:
    shard_id: int
    event_ids: np.ndarray
    summary_features: np.ndarray
    summary_quality_flags: np.ndarray
    primary_label_ids: np.ndarray
    ref_row_features: np.ndarray
    ref_row_event_index: np.ndarray
    ref_row_mask: np.ndarray
    event_ref_row_features: np.ndarray
    event_ref_row_mask: np.ndarray
    n_events: int
    n_ref_rows: int


@dataclass
class EvidenceBatch:
    event_ids: list[str]
    summary_features: "torch.Tensor"
    summary_quality_flags: "torch.Tensor"
    primary_label_ids: "torch.Tensor"
    ref_row_features: "torch.Tensor"
    ref_row_mask: "torch.Tensor"


@dataclass
class EvidenceEncoderOutput:
    summary_hidden: "torch.Tensor"
    reference_hidden: "torch.Tensor"
    event_hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
