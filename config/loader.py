from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.types import DATASET_IDS, VIEW_NAMES
from config.utils import deep_merge


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
REQUIRED_TOP_LEVEL_KEYS = {"meta", "stages"}


def _config_path(mode: str) -> Path:
    if mode != "full":
        raise KeyError(f"Unsupported config mode: {mode}")
    return CONFIG_PATH


def _read_config_file(mode: str) -> dict[str, Any]:
    path = _config_path(mode)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_TOP_LEVEL_KEYS - set(payload)
    if missing:
        raise KeyError(f"{path.name} missing required top-level keys: {sorted(missing)}")
    return payload


@lru_cache(maxsize=1)
def load_config(mode: str) -> dict[str, Any]:
    payload = _read_config_file(mode)
    config_mode = ((payload.get("meta") or {}).get("mode")) or mode
    if config_mode != mode:
        raise ValueError(f"{_config_path(mode).name} meta.mode={config_mode!r} does not match requested mode={mode!r}")
    stages = payload.get("stages", {})
    for stage_name in ("event_pipeline", "encoders", "fusion"):
        if stage_name not in stages:
            raise KeyError(f"{_config_path(mode).name} missing stage entry for {stage_name}")
    encoders = (stages.get("encoders") or {})
    for view_name in VIEW_NAMES:
        if view_name not in encoders:
            raise KeyError(f"{_config_path(mode).name} missing encoder stage entry for {view_name}")
    event_pipeline_overrides = (stages.get("event_pipeline", {}).get("dataset_overrides") or {})
    fusion_overrides = (stages.get("fusion", {}).get("dataset_overrides") or {})
    for dataset_id in DATASET_IDS:
        if dataset_id in event_pipeline_overrides and not isinstance(event_pipeline_overrides[dataset_id], dict):
            raise TypeError(f"event_pipeline.dataset_overrides[{dataset_id!r}] must be a dict")
        if dataset_id in fusion_overrides and not isinstance(fusion_overrides[dataset_id], dict):
            raise TypeError(f"fusion.dataset_overrides[{dataset_id!r}] must be a dict")
    return payload


def get_meta(mode: str) -> dict[str, Any]:
    return dict(load_config(mode).get("meta", {}))


def _stage_defaults(mode: str, stage: str, view_name: str | None = None) -> dict[str, Any]:
    stages = load_config(mode).get("stages", {})
    stage_payload = stages.get(stage, {})
    if stage == "encoders":
        if view_name is None:
            raise ValueError("view_name is required for encoder defaults")
        return dict((((stage_payload.get(view_name) or {}).get("defaults")) or {}))
    return dict((stage_payload.get("defaults")) or {})


def _dataset_stage_overrides(mode: str, dataset_id: str, stage: str, view_name: str | None = None) -> dict[str, Any]:
    stages = load_config(mode).get("stages", {})
    stage_payload = stages.get(stage, {})
    if stage == "encoders":
        view_payload = (stage_payload.get(view_name or "") or {})
        return dict((((view_payload.get("dataset_overrides")) or {}).get(dataset_id)) or {})
    return dict((((stage_payload.get("dataset_overrides")) or {}).get(dataset_id)) or {})


def resolve_stage_overrides(mode: str, dataset_id: str, stage: str, view_name: str | None = None) -> dict[str, Any]:
    return deep_merge(
        _stage_defaults(mode, stage, view_name),
        _dataset_stage_overrides(mode, dataset_id, stage, view_name),
    )
