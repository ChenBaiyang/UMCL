from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import write_array

from models.encoders.stance_view.schemas import StanceArrayShard
from utils.array_bundle import write_array_bundle


def _save_npz(path: Path, *, compresslevel: int = 1, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
        for key, value in arrays.items():
            array = np.asarray(value)
            with archive.open(f"{key}.npy", mode="w") as handle:
                write_array(handle, array, allow_pickle=False)
    write_array_bundle(path, arrays)


def write_stance_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stance_label_map(path: Path, payload: dict[str, Any]) -> None:
    write_stance_manifest(path, payload)
def write_stance_shard(path: Path, shard: StanceArrayShard) -> None:
    _save_npz(
        path,
        shard_id=np.asarray([shard.shard_id], dtype=np.int32),
        reply_embedding=np.asarray(shard.reply_embedding, dtype=np.float16),
        context_cat=np.asarray(shard.context_cat, dtype=np.int16),
        context_dense=np.asarray(shard.context_dense, dtype=np.float32),
        reply_label_ids=np.asarray(shard.reply_label_ids, dtype=np.int16),
        reply_label_mask=np.asarray(shard.reply_label_mask, dtype=np.uint8),
        event_index=np.asarray(shard.event_index, dtype=np.int32),
        event_sampled_reply_count=np.asarray(shard.event_sampled_reply_count, dtype=np.int16),
        event_truncation_flag=np.asarray(shard.event_truncation_flag, dtype=np.uint8),
    )
