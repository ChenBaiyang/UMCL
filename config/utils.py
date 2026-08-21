from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return config
    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass config, got {type(config)!r}")
    valid_fields = {field.name for field in fields(config)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise AttributeError(f"{type(config).__name__} has no configurable field {key!r}")
        setattr(config, key, copy.deepcopy(value))
    if "resolved_overrides" in valid_fields:
        config.resolved_overrides = copy.deepcopy(overrides)
    return config
