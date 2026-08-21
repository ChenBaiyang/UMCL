"""Canonical schema definitions for raw social-media credibility datasets.

This module is intentionally model-agnostic. It defines only the
raw observable records required by the canonical conversion layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetMeta:
    dataset_id: str
    platform: str
    task_family: str
    time_format: str
    label_spaces: dict[str, list[str]]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    event_id: str
    dataset_id: str
    platform: str
    source_post_id: str
    split: str
    topic: str | None
    lang: str | None
    start_time: str | None
    end_time: str | None
    view_status: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PostRecord:
    post_id: str
    event_id: str
    user_id: str | None
    post_type: str
    interaction_type: str
    text_status: str
    parent_post_id: str | None
    root_post_id: str
    created_at: str | None
    text_raw: str | None
    lang: str | None
    engagement: dict[str, Any]
    has_media: bool | None
    has_url: bool | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UserRecord:
    user_id: str
    platform: str
    account_created_at: str | None
    verified: bool | None
    followers_count: int | None
    following_count: int | None
    post_count: int | None
    profile_description_raw: str | None
    display_name_raw: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceRecord:
    ref_id: str
    event_id: str
    post_id: str
    ref_type: str
    raw_value: str | None
    normalized_value: str | None
    domain: str | None
    target_post_id: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationRecord:
    annotation_id: str
    target_type: str
    target_id: str
    task: str
    label: str
    label_space: str
    original_label: str
    anchor_post_id: str | None
    annotator_kind: str
    confidence: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
