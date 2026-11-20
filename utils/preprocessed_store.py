from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator


PREPROCESSED_EVENTS_FILENAME = "preprocessed_events.pkl.gz"


def _is_jsonl_path(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def write_preprocessed_events(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, mode="wb") as handle:
        for row in rows:
            pickle.dump(row, handle, protocol=pickle.HIGHEST_PROTOCOL)
            count += 1
    return count


def iter_preprocessed_events(path: Path) -> Iterator[dict[str, Any]]:
    if _is_jsonl_path(path):
        from utils.jsonl import iter_jsonl

        yield from iter_jsonl(path)
        return

    with gzip.open(path, mode="rb") as handle:
        while True:
            try:
                row = pickle.load(handle)
            except EOFError:
                break
            if row is not None:
                yield row


def read_preprocessed_events(path: Path) -> list[dict[str, Any]]:
    return list(iter_preprocessed_events(path))
