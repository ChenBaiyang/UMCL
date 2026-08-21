from __future__ import annotations

from typing import Any


def observation_kind_id(kind: str | None) -> int:
    return {"missing": 0, "absolute_timestamp": 1, "relative_delay_observed": 2}.get(kind or "missing", 0)


def time_bucket_id(node_time: dict[str, Any]) -> int:
    minutes = node_time.get("time_from_source_min")
    if minutes is None:
        values = node_time.get("time_delay_minutes_raw_values") or []
        if values:
            try:
                minutes = float(values[0])
            except (TypeError, ValueError):
                minutes = None
    if minutes is None:
        return 0
    if minutes <= 30:
        return 1
    if minutes <= 360:
        return 2
    return 3


def continuous_time_values(node_time: dict[str, Any]) -> tuple[float, float, int]:
    sec = node_time.get("time_from_source_sec")
    minute = node_time.get("time_from_source_min")
    if minute is None:
        values = node_time.get("time_delay_minutes_raw_values") or []
        if values:
            try:
                minute = float(values[0])
            except (TypeError, ValueError):
                minute = None
    if sec is None and minute is not None:
        sec = float(minute) * 60.0
    has_continuous = 1 if (sec is not None or minute is not None) else 0
    return float(sec or 0.0), float(minute or 0.0), has_continuous


def time_sort_key(node: dict[str, Any]) -> tuple[int, float, int, str]:
    node_time = node.get("time") or {}
    kind = node_time.get("observation_kind")
    created_at = node_time.get("created_at")
    if kind == "absolute_timestamp" and created_at:
        return (0, str(created_at), int(node.get("stable_order") or 0), str(node["post_id"]))
    sec, minute, has_continuous = continuous_time_values(node_time)
    if kind == "relative_delay_observed" and has_continuous:
        return (1, sec, int(node.get("stable_order") or 0), str(node["post_id"]))
    return (2, float(int(node.get("stable_order") or 10**9)), int(node.get("stable_order") or 10**9), str(node["post_id"]))
