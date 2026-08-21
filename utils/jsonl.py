from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Union, get_args, get_origin, get_type_hints


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: dataclass_to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dataclass_to_dict(row), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def dataclass_from_dict(cls: type[Any], payload: Any) -> Any:
    if payload is None:
        return None

    origin = get_origin(cls)
    if origin is Union:
        non_none = [arg for arg in get_args(cls) if arg is not type(None)]
        if not non_none:
            return None
        return dataclass_from_dict(non_none[0], payload)
    if origin is list:
        item_type = get_args(cls)[0]
        return [dataclass_from_dict(item_type, item) for item in payload]
    if origin is dict:
        key_type, value_type = get_args(cls)
        if value_type is Any:
            return dict(payload)
        return {key: dataclass_from_dict(value_type, value) for key, value in payload.items()}
    if is_dataclass(cls):
        hints = get_type_hints(cls)
        kwargs = {}
        for field_def in fields(cls):
            if field_def.name not in payload:
                continue
            kwargs[field_def.name] = dataclass_from_dict(hints[field_def.name], payload[field_def.name])
        return cls(**kwargs)
    return payload
