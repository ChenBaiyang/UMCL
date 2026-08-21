from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from models.encoders.propagation_view.schemas import PropagationArrayShard
from utils.array_bundle import write_array_bundle


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_array_bundle(path, arrays)


def write_propagation_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
def write_propagation_shard(path: Path, shard: PropagationArrayShard) -> None:
    _save_npz(
        path,
        shard_id=np.asarray([shard.shard_id], dtype=np.int32),
        event_ids=np.asarray(shard.event_ids, dtype=object),
        node_post_ids=np.asarray(shard.node_post_ids, dtype=object),
        node_features_continuous=np.asarray(shard.node_features_continuous, dtype=np.float32),
        node_features_categorical=np.asarray(shard.node_features_categorical, dtype=np.int16),
        node_masks=np.asarray(shard.node_masks, dtype=np.uint8),
        parent_index=np.asarray(shard.parent_index, dtype=np.int32),
        node_sampling_role_ids=np.asarray(shard.node_sampling_role_ids, dtype=np.int16),
        event_index=np.asarray(shard.event_index, dtype=np.int32),
        label_ids=np.asarray(shard.label_ids, dtype=np.int16),
        label_mask=np.asarray(shard.label_mask, dtype=np.uint8),
        event_summary_features=np.asarray(shard.event_summary_features, dtype=np.float32),
        temporal_profile_features=np.asarray(shard.temporal_profile_features, dtype=np.float32),
        max_nodes_per_event=np.asarray([shard.max_nodes_per_event], dtype=np.int32),
        n_node_examples=np.asarray([shard.n_node_examples], dtype=np.int32),
    )
