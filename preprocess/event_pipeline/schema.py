from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Platform = Literal["twitter", "weibo", "reddit", "unknown"]
Split = Literal["train", "val", "dev", "test", "all", "unknown"]
ViewName = Literal["content", "stance", "propagation", "evidence", "source"]

InteractionType = Literal[
    "source",
    "reply",
    "repost",
    "quote_repost",
    "unknown",
]

TextStatus = Literal[
    "substantive",
    "boilerplate",
    "empty",
    "placeholder",
    "unknown",
]

StanceLabel = Literal[
    "support",
    "deny",
    "query",
    "comment",
]

AnnotatorKind = Literal[
    "gold",
    "silver",
    "proxy",
    "mapped",
    "none",
]

ReferenceType = Literal[
    "url",
    "quoted_post",
    "reposted_post",
    "media",
]

TimeObservationKind = Literal[
    "absolute_timestamp",
    "relative_delay_observed",
    "missing",
]

SourceValueStatus = Literal[
    "observed",
    "missing",
]

TaskName = Literal[
    "veracity",
    "rumour_status",
    "stance",
    "unknown",
]


@dataclass
class TextQualityFlags:
    is_text_missing: bool = False
    is_text_empty: bool = False
    is_text_placeholder: bool = False
    is_text_boilerplate: bool = False
    is_text_substantive: bool = False
    is_semantically_usable: bool = False


@dataclass
class TextBundle:
    raw: Optional[str] = None
    norm_light: Optional[str] = None
    lang: Optional[str] = None
    status: TextStatus = "unknown"
    quality: TextQualityFlags = field(default_factory=TextQualityFlags)
    char_len: Optional[int] = None
    token_len_approx: Optional[int] = None


@dataclass
class TimeBundle:
    observation_kind: TimeObservationKind = "missing"
    created_at: Optional[str] = None
    time_from_source_sec: Optional[float] = None
    time_from_source_min: Optional[float] = None
    time_delay_minutes_raw_values: Optional[List[float]] = None
    time_delay_minutes_bucketed: Optional[List[str]] = None
    time_delay_hours_bucketed: Optional[List[str]] = None
    log_time_from_source_sec: Optional[float] = None
    is_observed_time: bool = False


@dataclass
class EngagementBundle:
    like_count: Optional[int] = None
    reply_count: Optional[int] = None
    repost_count: Optional[int] = None


@dataclass
class TaskTarget:
    task: TaskName
    label: Optional[str]
    label_space: Optional[str] = None
    original_label: Optional[str] = None
    annotator_kind: AnnotatorKind = "none"
    confidence: Optional[float] = None


@dataclass
class ViewAvailability:
    observed: bool
    source_type: str
    notes: List[str] = field(default_factory=list)


@dataclass
class ContentSurfaceMeta:
    has_url: Optional[bool] = None
    has_media: Optional[bool] = None
    engagement: EngagementBundle = field(default_factory=EngagementBundle)
    platform_extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentView:
    source_post_id: str
    source_user_id: Optional[str]
    interaction_type: InteractionType = "source"
    text: TextBundle = field(default_factory=TextBundle)
    time: TimeBundle = field(default_factory=TimeBundle)
    surface: ContentSurfaceMeta = field(default_factory=ContentSurfaceMeta)


@dataclass
class ReplyItem:
    post_id: str
    parent_post_id: Optional[str]
    user_id: Optional[str]
    interaction_type: InteractionType = "reply"
    text: TextBundle = field(default_factory=TextBundle)
    time: TimeBundle = field(default_factory=TimeBundle)
    stance_label: Optional[StanceLabel] = None
    stance_label_source: AnnotatorKind = "none"
    is_placeholder: bool = False
    is_observed_text: bool = False
    is_semantically_substantive: bool = False
    is_boilerplate_text: bool = False
    depth: Optional[int] = None
    parent_is_source: Optional[bool] = None
    stable_order: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StanceSummary:
    n_replies_total: int = 0
    n_replies_text_observed: int = 0
    n_replies_semantically_substantive: int = 0
    n_replies_stance_labeled: int = 0
    n_placeholders: int = 0
    n_boilerplate_replies: int = 0


@dataclass
class StanceView:
    source_post_id: str
    reply_items: List[ReplyItem] = field(default_factory=list)
    summary: StanceSummary = field(default_factory=StanceSummary)


@dataclass
class PropagationNode:
    post_id: str
    parent_post_id: Optional[str]
    is_source: bool = False
    interaction_type: InteractionType = "reply"
    text_status: TextStatus = "unknown"
    is_placeholder: bool = False
    is_observed_text: bool = False
    time: TimeBundle = field(default_factory=TimeBundle)
    depth: Optional[int] = None
    stable_order: Optional[int] = None
    parent_inferred: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationEdge:
    parent_post_id: str
    child_post_id: str


@dataclass
class PropagationSummary:
    n_posts: int = 0
    n_replies: int = 0
    max_depth: Optional[int] = None
    n_observed_timestamps: int = 0
    first_reply_time_sec: Optional[float] = None
    n_placeholders: int = 0
    n_text_unobserved_nodes: int = 0
    n_boilerplate_nodes: int = 0


@dataclass
class PropagationView:
    root_post_id: str
    source_post_id: str
    source_time: TimeBundle = field(default_factory=TimeBundle)
    nodes: List[PropagationNode] = field(default_factory=list)
    edges: List[PropagationEdge] = field(default_factory=list)
    summary: PropagationSummary = field(default_factory=PropagationSummary)


@dataclass
class EvidenceReference:
    ref_id: str
    ref_type: ReferenceType
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    domain: Optional[str] = None
    target_post_id: Optional[str] = None
    is_internal_platform_ref: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSummary:
    scope: Literal["source_only"] = "source_only"
    has_explicit_ref: bool = False
    n_refs_total: int = 0
    n_url_refs: int = 0
    n_media_refs: int = 0
    n_internal_refs: int = 0
    n_external_refs: int = 0
    thread_n_refs_total: Optional[int] = None


@dataclass
class EvidenceView:
    source_post_id: str
    references: List[EvidenceReference] = field(default_factory=list)
    summary: EvidenceSummary = field(default_factory=EvidenceSummary)


@dataclass
class SourceMissingness:
    account_created_at: SourceValueStatus = "missing"
    verified: SourceValueStatus = "missing"
    followers_count: SourceValueStatus = "missing"
    following_count: SourceValueStatus = "missing"
    post_count: SourceValueStatus = "missing"
    profile_description_raw: SourceValueStatus = "missing"
    display_name_raw: SourceValueStatus = "missing"


@dataclass
class SourceProfile:
    account_created_at: Optional[str] = None
    verified: Optional[bool] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    profile_description: TextBundle = field(default_factory=TextBundle)
    display_name: TextBundle = field(default_factory=TextBundle)
    account_age_days: Optional[float] = None
    missingness: SourceMissingness = field(default_factory=SourceMissingness)
    platform_extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceView:
    source_user_id: Optional[str]
    profile: SourceProfile = field(default_factory=SourceProfile)


@dataclass
class UnifiedEvent:
    event_id: str
    dataset_id: str
    platform: Platform
    canonical_split: Split = "unknown"
    raw_split_source: Optional[str] = None
    task_targets: Dict[str, TaskTarget] = field(default_factory=dict)
    observed_view_mask: Dict[ViewName, int] = field(default_factory=dict)
    view_availability: Dict[ViewName, ViewAvailability] = field(default_factory=dict)
    content: Optional[ContentView] = None
    stance: Optional[StanceView] = None
    propagation: Optional[PropagationView] = None
    evidence: Optional[EvidenceView] = None
    source: Optional[SourceView] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


from config.event_pipeline import EventPipelineConfig as PipelineConfig
