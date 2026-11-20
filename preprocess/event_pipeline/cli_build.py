from __future__ import annotations

import argparse
import csv
import gc
import gzip
import heapq
import math
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.event_pipeline.event_builder import UnifiedEventBuilder
from preprocess.event_pipeline.id_registry import IdCompactor
from preprocess.event_pipeline.preprocessing import preprocess_unified_event
from preprocess.event_pipeline.dataset_ingest import available_dataset_ids, open_raw_dataset
from config.resolver import resolve_event_pipeline_config
from preprocess.event_pipeline.validation import validate_unified_event
from utils.jsonl import dataclass_to_dict, write_json
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME
from utils.logging_utils import get_logger


LOGGER = get_logger("event_pipeline_build")
GZIP_COMPRESSLEVEL = 1
PROGRESS_LOG_EVERY = 250
RATIO_SPLITS = ("train", "val", "test")
GENERIC_SPLIT_RATIOS = {"train": 70, "val": 15, "test": 15}


@dataclass(frozen=True)
class SplitAssignment:
    canonical_split: str
    raw_split_source: str | None


def _stringify_split(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return "unknown"
    return str(value)


def _manifest_text(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return ""
    return str(value).strip()


def _original_split_source(event_row: dict[str, object]) -> str | None:
    raw_source = event_row.get("raw_split_source")
    if raw_source is not None:
        return _stringify_split(raw_source)
    split_value = event_row.get("split")
    if split_value is None:
        split_value = event_row.get("canonical_split")
    canonical = _stringify_split(split_value)
    return None if canonical == "unknown" else canonical


def _raw_event_thread_id(event_row: dict[str, object]) -> str:
    extra = event_row.get("extra")
    if isinstance(extra, dict):
        for key in ("event_id_raw", "raw_event_id", "thread_id_raw", "thread_id"):
            value = extra.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        thread_path = extra.get("thread_path")
        if thread_path is not None and str(thread_path).strip():
            path = Path(str(thread_path))
            return path.stem if path.suffix else path.name
    return str(event_row.get("event_id") or "")


def _raw_source_post_id(event_row: dict[str, object], dataset: object) -> str:
    source_post_id = event_row.get("source_post_id")
    source_post = None
    if source_post_id is not None and hasattr(dataset, "get"):
        source_post = dataset.get("posts", source_post_id)
    if isinstance(source_post, dict):
        extra = source_post.get("extra")
        if isinstance(extra, dict):
            for key in ("source_post_id_raw", "post_id_raw", "tweet_id_raw", "raw_post_id"):
                value = extra.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        post_id = source_post.get("post_id")
        if post_id is not None:
            post_id_str = str(post_id).strip()
            if post_id_str:
                return post_id_str.rsplit(":", 1)[-1]
    if source_post_id is not None:
        source_post_id_str = str(source_post_id).strip()
        if source_post_id_str:
            return source_post_id_str.rsplit(":", 1)[-1]
    return ""


def _split_task_name(dataset_id: str) -> str:
    return "rumour_status" if dataset_id == "Weibo" else "veracity"


def _build_split_label_map(dataset_id: str, annotation_rows: list[dict[str, object]]) -> dict[str, str]:
    target_task = _split_task_name(dataset_id)
    label_map: dict[str, str] = {}
    for annotation in annotation_rows:
        if annotation.get("target_type") != "event":
            continue
        task = str(annotation.get("task") or "")
        if task != target_task:
            continue
        label = annotation.get("label")
        if label is None:
            continue
        label_map[str(annotation.get("target_id"))] = str(label).strip().lower()
    return label_map


def _split_label_name(dataset_id: str, event_row: dict[str, object], label_map: dict[str, str]) -> str:
    event_id = str(event_row.get('event_id'))
    label = label_map.get(event_id)
    if label is not None:
        return label.strip().lower()
    task_targets = event_row.get('task_targets') or {}
    preferred_tasks = [_split_task_name(dataset_id)]
    if dataset_id == 'Weibo':
        preferred_tasks.append('veracity')
    for task_name in preferred_tasks:
        target = task_targets.get(task_name) if isinstance(task_targets, dict) else None
        if isinstance(target, dict):
            label = target.get('label')
            if label is not None:
                return str(label).strip().lower()
    if isinstance(task_targets, dict):
        for target in task_targets.values():
            if isinstance(target, dict):
                label = target.get('label')
                if label is not None:
                    return str(label).strip().lower()
    raise ValueError(f"Unable to resolve split label for dataset={dataset_id} event_id={event_row.get('event_id')!r}")


def _target_split_counts(total_count: int, dataset_id: str) -> dict[str, int]:
    split_names = list(RATIO_SPLITS)
    ratios = [GENERIC_SPLIT_RATIOS[name] for name in split_names]
    scaled = [total_count * ratio / sum(ratios) for ratio in ratios]
    counts = [math.floor(value) for value in scaled]
    remainder = total_count - sum(counts)
    order = sorted(range(len(split_names)), key=lambda idx: (scaled[idx] - counts[idx], ratios[idx], -idx), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return {split_names[idx]: int(counts[idx]) for idx in range(len(split_names))}


def _write_split_manifest(
    *,
    dataset_id: str,
    split_seed: int,
    downloads_root: str | Path,
    event_rows: list[dict[str, object]],
    split_assignments: dict[str, SplitAssignment],
    label_map: dict[str, str],
    dataset: object,
) -> Path:
    manifest_path = Path(downloads_root) / f"{dataset_id}_split_manifest_seed_{split_seed}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for event_row in event_rows:
        canonical_event_id = str(event_row["event_id"])
        assignment = split_assignments[canonical_event_id]
        rows.append(
            {
                "seed_id": split_seed,
                "thread_id": _manifest_text(_raw_event_thread_id(event_row)),
                "source_post_id": _manifest_text(_raw_source_post_id(event_row, dataset)),
                "label": _split_label_name(dataset_id, event_row, label_map),
                "split": assignment.canonical_split,
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed_id", "thread_id", "source_post_id", "label", "split"],
            quoting=csv.QUOTE_NONNUMERIC,
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _choose_split(remaining: dict[str, int], target: dict[str, int], *, preferred_order: tuple[str, ...]) -> str:
    candidates = [name for name, quota in remaining.items() if quota > 0]
    if not candidates:
        raise ValueError("No split quota remaining.")
    def _score(split_name: str) -> tuple[float, int, int]:
        target_count = max(1, target[split_name])
        return (remaining[split_name] / target_count, remaining[split_name], -preferred_order.index(split_name))
    return max(candidates, key=_score)


def _build_split_assignments(dataset_id: str, event_rows: list[dict[str, object]], annotation_rows: list[dict[str, object]], split_seed: int, *, label_map: dict[str, str] | None=None) -> dict[str, SplitAssignment]:
    rng = random.Random(int(split_seed))
    label_map = label_map or _build_split_label_map(dataset_id, annotation_rows)
    assignments: dict[str, SplitAssignment] = {}
    target_counts = _target_split_counts(len(event_rows), dataset_id)
    assignments.update(_stratified_assignments(event_rows, target_counts, label_map=label_map, preferred_order=RATIO_SPLITS, rng=rng))
    return assignments


def _stratified_assignments(
    event_rows: list[dict[str, object]],
    target_counts: dict[str, int],
    *,
    label_map: dict[str, str],
    preferred_order: tuple[str, ...],
    rng: random.Random,
) -> dict[str, SplitAssignment]:
    label_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        label_buckets[_split_label_name(str(row.get("dataset_id")), row, label_map)].append(row)

    for bucket in label_buckets.values():
        rng.shuffle(bucket)

    class_heap: list[tuple[int, str]] = [(-len(bucket), label) for label, bucket in label_buckets.items()]
    heapq.heapify(class_heap)
    remaining = dict(target_counts)
    assignments: dict[str, SplitAssignment] = {}

    while class_heap:
        _, label = heapq.heappop(class_heap)
        bucket = label_buckets[label]
        row = bucket.pop()
        split_name = _choose_split(remaining, target_counts, preferred_order=preferred_order)
        event_id = str(row["event_id"])
        assignments[event_id] = SplitAssignment(canonical_split=split_name, raw_split_source=_original_split_source(row))
        remaining[split_name] -= 1
        if bucket:
            heapq.heappush(class_heap, (-len(bucket), label))

    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"Split allocation did not exhaust quotas: {remaining}")
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preprocessed UnifiedEvent rows directly from raw downloads."
    )
    parser.add_argument("--dataset_id", required=True, choices=available_dataset_ids())
    parser.add_argument("--downloads_root", default="data/downloads")
    parser.add_argument("--output_root", default="data/preprocessed")
    parser.add_argument("--id_map_mode", default="summary", choices=["summary", "full", "none"])
    parser.add_argument("--weibo_converter_workers", type=int, default=None)
    parser.add_argument("--progress_log_every", type=int, default=PROGRESS_LOG_EVERY)
    return parser.parse_args()


def main() -> None:
    from preprocess.preprocess_events import preprocess_events

    preprocess_events(parse_args())


if __name__ == "__main__":
    main()
