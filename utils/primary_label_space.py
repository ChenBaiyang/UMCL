from __future__ import annotations

from pathlib import Path
from typing import Any

NON_RUMOUR_CANONICAL = "non-rumour"
NON_RUMOUR_LEGACY = "non_rumour"
MISSING_LABEL_ID = -1


def _normalize_label_name(label: str | None) -> str | None:
    if label is None:
        return None
    if label == NON_RUMOUR_LEGACY:
        return NON_RUMOUR_CANONICAL
    return str(label)


def resolve_primary_task_spec(dataset_id: str, data_root: str | Path='data') -> dict[str, Any]:
    _ = data_root
    dataset_id = str(dataset_id)
    specs: dict[str, dict[str, Any]] = {'PHEME': {'task_name': 'veracity', 'metric_name': 'veracity_4way', 'class_names': ['true', 'false', 'unverified', 'non-rumour']}, 'Twitter15': {'task_name': 'veracity', 'metric_name': 'veracity_4way', 'class_names': ['true', 'false', 'unverified', 'non-rumour']}, 'Twitter16': {'task_name': 'veracity', 'metric_name': 'veracity_4way', 'class_names': ['true', 'false', 'unverified', 'non-rumour']}, 'Weibo': {'task_name': 'rumour_status', 'metric_name': 'rumour_status_2way', 'class_names': ['rumour', 'non-rumour']}}
    if dataset_id not in specs:
        raise ValueError(f'Dataset {dataset_id} has no supported primary label-space specification.')
    spec = specs[dataset_id]
    class_names = [_normalize_label_name(name) for name in spec['class_names']]
    return {'dataset_id': dataset_id, 'task_name': spec['task_name'], 'metric_name': spec['metric_name'], 'class_names': class_names, 'class_name_to_id': {name: idx for (idx, name) in enumerate(class_names)}}


def align_primary_label(
    dataset_id: str,
    task_targets: dict[str, Any],
    data_root: str | Path = "data",
) -> dict[str, Any]:
    spec = resolve_primary_task_spec(dataset_id, data_root)
    task_name = str(spec["task_name"])
    target = task_targets.get(task_name) or {}
    label_name = _normalize_label_name(target.get("label"))
    label_id = int(spec["class_name_to_id"].get(label_name, MISSING_LABEL_ID))
    return {
        "primary_task_name": task_name,
        "primary_metric_name": spec["metric_name"],
        "primary_class_names": list(spec["class_names"]),
        "primary_label_name": label_name,
        "primary_label_id": label_id,
    }


def primary_task_spec_from_index_payload(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task_name = _normalize_label_name(_scalar(payload["primary_task_name"]))
    metric_name = _normalize_label_name(_scalar(payload["primary_metric_name"]))
    class_names = [_normalize_label_name(name) for name in payload["primary_class_names"].tolist()]
    return {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "metric_name": metric_name,
        "class_names": class_names,
        "class_name_to_id": {name: idx for idx, name in enumerate(class_names)},
    }


def _scalar(value: Any) -> Any:
    if hasattr(value, "shape") and getattr(value, "shape", ()) == (1,):
        value = value[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
