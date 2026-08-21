from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from models.encoders.evidence_view.schemas import EvidenceArrayShard
from utils.array_bundle import write_array_bundle
from utils.jsonl import write_jsonl


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_array_bundle(path, arrays)


def write_evidence_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_evidence_event_meta(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def write_evidence_reference_meta(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)
def write_evidence_shard(path: Path, shard: EvidenceArrayShard) -> None:
    _save_npz(
        path,
        shard_id=np.asarray([shard.shard_id], dtype=np.int32),
        event_ids=np.asarray(shard.event_ids, dtype=object),
        summary_features=np.asarray(shard.summary_features, dtype=np.float32),
        summary_quality_flags=np.asarray(shard.summary_quality_flags, dtype=np.uint8),
        primary_label_ids=np.asarray(shard.primary_label_ids, dtype=np.int32),
        ref_row_features=np.asarray(shard.ref_row_features, dtype=np.float32),
        ref_row_event_index=np.asarray(shard.ref_row_event_index, dtype=np.int32),
        ref_row_mask=np.asarray(shard.ref_row_mask, dtype=np.uint8),
        event_ref_row_features=np.asarray(shard.event_ref_row_features, dtype=np.float32),
        event_ref_row_mask=np.asarray(shard.event_ref_row_mask, dtype=np.uint8),
        n_events=np.asarray([shard.n_events], dtype=np.int32),
        n_ref_rows=np.asarray([shard.n_ref_rows], dtype=np.int32),
    )
