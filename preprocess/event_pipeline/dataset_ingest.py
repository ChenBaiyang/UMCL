from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from preprocess.canonical.converters import (
    PhemeVeracityCanonicalConverter,
    RumdectWeiboCanonicalConverter,
    RumourEvalCanonicalConverter,
    Twitter15CanonicalConverter,
    Twitter16CanonicalConverter,
)
from preprocess.event_pipeline.raw_archive import open_download_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_ROOT = PROJECT_ROOT / "data" / "downloads"


def available_dataset_ids() -> list[str]:
    return ["RumourEval", "PHEME", "Twitter15", "Twitter16", "Weibo"]


def _converter_specs(download_root: Path) -> dict[str, tuple[Path, Callable[[Path, Path], object]]]:
    return {
        "RumourEval": (download_root / "RumourEval", RumourEvalCanonicalConverter),
        "PHEME": (download_root / "PHEME", PhemeVeracityCanonicalConverter),
        "Twitter15": (download_root / "Twitter15", Twitter15CanonicalConverter),
        "Twitter16": (download_root / "Twitter16", Twitter16CanonicalConverter),
        "Weibo": (download_root / "Weibo", RumdectWeiboCanonicalConverter),
    }


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def _row_field(row: Any, field_name: str) -> Any:
    if isinstance(row, dict):
        return row.get(field_name)
    return getattr(row, field_name)


class InMemoryCanonicalDataset:
    def __init__(
        self,
        *,
        dataset_id: str,
        events: list[dict[str, Any]],
        posts: list[dict[str, Any]],
        users: list[dict[str, Any]],
        references: list[dict[str, Any]],
        annotations: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
        report: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.meta = meta or {}
        self.report = report or {}
        self._events = events if isinstance(events, list) else list(events)
        self._posts = posts if isinstance(posts, list) else list(posts)
        self._users = users if isinstance(users, list) else list(users)
        self._references = references if isinstance(references, list) else list(references)
        self._annotations = annotations if isinstance(annotations, list) else list(annotations)
        self._row_cache: dict[int, dict[str, Any]] = {}

        self._events_by_id = {_row_field(row, "event_id"): row for row in self._events}
        self._posts_by_id = {_row_field(row, "post_id"): row for row in self._posts}
        self._users_by_id = {_row_field(row, "user_id"): row for row in self._users}
        self._references_by_id = {_row_field(row, "ref_id"): row for row in self._references}
        self._annotations_by_id = {_row_field(row, "annotation_id"): row for row in self._annotations}
        self._events_sorted = self._events
        self._posts_sorted = self._posts
        self._references_sorted = self._references
        self._annotations_sorted = self._annotations
        self._posts_by_event: dict[Any, list[Any]] = {}
        self._references_by_event: dict[Any, list[Any]] = {}
        self._references_by_post: dict[Any, list[Any]] = {}
        for row in self._posts:
            self._posts_by_event.setdefault(_row_field(row, "event_id"), []).append(row)
        for row in self._references:
            self._references_by_event.setdefault(_row_field(row, "event_id"), []).append(row)
            self._references_by_post.setdefault(_row_field(row, "post_id"), []).append(row)

    def _row_dict(self, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        cache_key = id(row)
        cached = self._row_cache.get(cache_key)
        if cached is None:
            cached = _row_to_dict(row)
            self._row_cache[cache_key] = cached
        return cached

    def __enter__(self) -> "InMemoryCanonicalDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def close(self) -> None:
        self._events.clear()
        self._posts.clear()
        self._users.clear()
        self._references.clear()
        self._annotations.clear()
        self._events_by_id.clear()
        self._posts_by_id.clear()
        self._users_by_id.clear()
        self._references_by_id.clear()
        self._annotations_by_id.clear()
        self._posts_by_event.clear()
        self._references_by_event.clear()
        self._references_by_post.clear()
        self._row_cache.clear()
        self._events_sorted = []
        self._posts_sorted = []
        self._references_sorted = []
        self._annotations_sorted = []

    def get(self, table: str, record_id: Any) -> dict[str, Any] | None:
        key = int(record_id) if isinstance(record_id, bool) is False and isinstance(record_id, int) else record_id
        if table == "events":
            row = self._events_by_id.get(key)
            return None if row is None else self._row_dict(row)
        if table == "posts":
            row = self._posts_by_id.get(key)
            return None if row is None else self._row_dict(row)
        if table == "users":
            row = self._users_by_id.get(key)
            return None if row is None else self._row_dict(row)
        if table == "references":
            row = self._references_by_id.get(key)
            return None if row is None else self._row_dict(row)
        if table == "annotations":
            row = self._annotations_by_id.get(key)
            return None if row is None else self._row_dict(row)
        raise KeyError(f"Unknown logical table: {table}")

    def iter_events(self) -> Iterable[dict[str, Any]]:
        for row in self._events_sorted:
            yield self._row_dict(row)

    def iter_posts(
        self,
        *,
        event_id: Any | None = None,
        user_id: Any | None = None,
        root_post_id: Any | None = None,
        limit: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        rows = self._posts_sorted
        if event_id is not None:
            rows = self._posts_by_event.get(event_id, [])
        elif user_id is not None or root_post_id is not None:
            rows = self._posts
        if event_id is not None and user_id is not None:
            rows = [row for row in rows if _row_field(row, "user_id") == user_id]
        if event_id is not None and root_post_id is not None:
            rows = [row for row in rows if _row_field(row, "root_post_id") == root_post_id]
        if event_id is None and user_id is not None and root_post_id is not None:
            rows = [row for row in rows if _row_field(row, "root_post_id") == root_post_id]
        if limit is not None:
            rows = rows[: int(limit)]
        for row in rows:
            yield self._row_dict(row)

    def iter_references(
        self,
        *,
        event_id: Any | None = None,
        post_id: Any | None = None,
        limit: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        rows = self._references_sorted
        if event_id is not None:
            rows = self._references_by_event.get(event_id, [])
        elif post_id is not None:
            rows = self._references_by_post.get(post_id, [])
        if event_id is not None and post_id is not None:
            rows = [row for row in rows if _row_field(row, "post_id") == post_id]
        if limit is not None:
            rows = rows[: int(limit)]
        for row in rows:
            yield self._row_dict(row)

    def iter_annotations(self) -> Iterable[dict[str, Any]]:
        for row in self._annotations_sorted:
            yield self._row_dict(row)


def open_raw_dataset(
    dataset_id: str,
    downloads_root: str | Path | None = None,
) -> InMemoryCanonicalDataset:
    root = Path(downloads_root) if downloads_root is not None else DEFAULT_DOWNLOAD_ROOT
    specs = _converter_specs(root)
    if dataset_id not in specs:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Expected one of {sorted(specs)}")
    _, converter_cls = specs[dataset_id]

    dummy_output_root = PROJECT_ROOT / ".cache" / "event_pipeline" / dataset_id
    with open_download_root(root, dataset_id) as resolved_input_root:
        converter = converter_cls(resolved_input_root, dummy_output_root)
        converter._write_outputs = lambda meta, report: None  # type: ignore[attr-defined]
        report = converter.convert()
        meta = getattr(converter, "meta", None)
        events = list(converter.events)
        posts = list(converter.posts_by_id.values())
        users = list(converter.users_by_id.values())
        references = list(converter.references)
        annotations = list(converter.annotations)
        converter.events = []
        converter.posts_by_id.clear()
        converter.users_by_id.clear()
        converter.references = []
        converter.annotations = []
        del converter
    shutil.rmtree(dummy_output_root, ignore_errors=True)
    dummy_parent = dummy_output_root.parent
    if dummy_parent.exists() and not any(dummy_parent.iterdir()):
        dummy_parent.rmdir()
    return InMemoryCanonicalDataset(
        dataset_id=dataset_id,
        events=events,
        posts=posts,
        users=users,
        references=references,
        annotations=annotations,
        meta=_row_to_dict(meta) if meta is not None else None,
        report=report,
    )
