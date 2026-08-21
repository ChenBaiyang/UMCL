from __future__ import annotations

from pathlib import Path

from preprocess.event_pipeline.schema import UnifiedEvent
from utils.jsonl import dataclass_from_dict, iter_jsonl, write_jsonl


def write_unified_events(path: Path, events: list[UnifiedEvent]) -> int:
    return write_jsonl(path, events)


def iter_unified_events(path: Path):
    for row in iter_jsonl(path):
        yield dataclass_from_dict(UnifiedEvent, row)


def read_unified_events(path: Path) -> list[UnifiedEvent]:
    return list(iter_unified_events(path))
