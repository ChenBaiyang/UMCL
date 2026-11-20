from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from config.event_pipeline import EventPipelineConfig
from config.loader import resolve_stage_overrides
from config.types import Mode, ViewName
from config.utils import apply_overrides
from models.encoders.content_view.config import ContentViewConfig
from models.encoders.evidence_view.config import EvidenceViewConfig
from models.encoders.propagation_view.config import PropagationViewConfig
from models.encoders.source_view.config import SourceViewConfig
from models.encoders.stance_view.config import StanceViewConfig
from models.fusion.config import FusionConfig


ENCODER_CONFIG_CLASSES: dict[str, type] = {
    "content": ContentViewConfig,
    "stance": StanceViewConfig,
    "propagation": PropagationViewConfig,
    "evidence": EvidenceViewConfig,
    "source": SourceViewConfig,
}
LEGACY_FUSION_OVERRIDE_KEYS = {
    "reply_count_bucket_edges",
    "propagation_node_count_bucket_edges",
}
PRETRAINED_MODEL_VIEWS = frozenset({"content", "stance"})
_PRETRAINED_MODEL_OVERRIDE: ContextVar[str | None] = ContextVar(
    "pretrained_model_override",
    default=None,
)


def set_pretrained_model_override(model_name: str | None) -> Token:
    """Temporarily override the HuggingFace model used by text encoders."""
    if model_name is not None:
        model_name = str(model_name).strip()
        if not model_name:
            raise ValueError("pretrained_model must not be empty.")
    return _PRETRAINED_MODEL_OVERRIDE.set(model_name)


def reset_pretrained_model_override(token: Token) -> None:
    _PRETRAINED_MODEL_OVERRIDE.reset(token)


def resolve_event_pipeline_config(dataset_id: str, mode: Mode = "full") -> EventPipelineConfig:
    config = EventPipelineConfig(dataset_id=dataset_id, mode=mode)
    overrides = resolve_stage_overrides(mode, dataset_id, "event_pipeline")
    return apply_overrides(config, overrides)


def resolve_encoder_config(view_name: ViewName | str, dataset_id: str, mode: Mode = "full") -> Any:
    view_name = str(view_name)
    config = ENCODER_CONFIG_CLASSES[view_name](dataset_id=dataset_id)
    overrides = resolve_stage_overrides(mode, dataset_id, "encoders", view_name)
    pretrained_model = _PRETRAINED_MODEL_OVERRIDE.get()
    if pretrained_model is not None and view_name in PRETRAINED_MODEL_VIEWS:
        overrides = {**overrides, "model_name_value": pretrained_model}
    return apply_overrides(config, overrides)


def resolve_fusion_config(dataset_id: str, mode: Mode = "full") -> FusionConfig:
    config = FusionConfig(dataset_id=dataset_id, mode=mode)
    overrides = resolve_stage_overrides(mode, dataset_id, "fusion")
    for legacy_key in LEGACY_FUSION_OVERRIDE_KEYS:
        overrides.pop(legacy_key, None)
    return apply_overrides(config, overrides)
