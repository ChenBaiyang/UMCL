from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.types import DATASET_IDS
from config.utils import deep_merge


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def _config_path(mode: str) -> Path:
    if mode != "full":
        raise KeyError(f"Unsupported config mode: {mode}")
    return CONFIG_PATH




@lru_cache(maxsize=1)
def load_config(mode: str) -> dict[str, Any]:
    payload = json.loads(_config_path(mode).read_text(encoding="utf-8"))
    if payload["meta"]["mode"] != mode:
        raise ValueError(f"Configuration mode must be {mode!r}.")
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
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"Unsupported dataset: {dataset_id!r}. Expected one of {DATASET_IDS}.")
    return deep_merge(
        _stage_defaults(mode, stage, view_name),
        _dataset_stage_overrides(mode, dataset_id, stage, view_name),
    )
