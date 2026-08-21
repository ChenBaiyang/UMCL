from __future__ import annotations

from typing import Any

from utils.primary_label_space import align_primary_label, primary_task_spec_from_index_payload, resolve_primary_task_spec

__all__ = ["align_primary_label", "resolve_primary_task_spec", "primary_task_spec_from_event_index", "primary_task_spec_from_content_index"]


def primary_task_spec_from_event_index(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return primary_task_spec_from_index_payload(dataset_id, payload)


def primary_task_spec_from_content_index(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return primary_task_spec_from_event_index(dataset_id, payload)
