from __future__ import annotations

from pathlib import Path

import numpy as np

from models.encoders.content_view.schemas import ContentArrayShard, ContentIndexRecord
from utils.array_bundle import write_array_bundle


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_array_bundle(path, arrays)


def write_content_index(
    path: Path,
    records: list[ContentIndexRecord],
    config_json: str,
    surface_feature_names: list[str],
    primary_class_names: list[str],
) -> None:
    arrays = {
        "event_ids": np.asarray([record.event_id for record in records], dtype=object),
        "dataset_ids": np.asarray([record.dataset_id for record in records], dtype=object),
        "platforms": np.asarray([record.platform for record in records], dtype=object),
        "canonical_splits": np.asarray([record.canonical_split for record in records], dtype=object),
        "raw_split_sources": np.asarray([record.raw_split_source for record in records], dtype=object),
        "primary_task_name": np.asarray([records[0].primary_task_name if records else None], dtype=object),
        "primary_metric_name": np.asarray([records[0].primary_metric_name if records else None], dtype=object),
        "primary_class_names": np.asarray(primary_class_names, dtype=object),
        "primary_label_names": np.asarray([record.primary_label_name for record in records], dtype=object),
        "primary_label_ids": np.asarray([record.primary_label_id for record in records], dtype=np.int16),
        "encoder_text_length_chars": np.asarray([record.encoder_text_length_chars for record in records], dtype=np.int32),
        "encoder_text_length_tokens_approx": np.asarray([record.encoder_text_length_tokens_approx for record in records], dtype=np.int32),
        "quality_flags": np.asarray(
            [
                [
                    int(record.is_text_missing),
                    int(record.is_text_empty),
                    int(record.is_text_boilerplate),
                    int(record.is_text_substantive),
                    int(record.is_semantically_usable),
                ]
                for record in records
            ],
            dtype=np.uint8,
        ),
        "has_url": np.asarray([record.has_url for record in records], dtype=np.uint8),
        "has_media": np.asarray([record.has_media for record in records], dtype=np.uint8),
        "surface_feature_dim": np.asarray([record.surface_feature_dim for record in records], dtype=np.int16),
        "shard_ids": np.asarray([record.shard_id for record in records], dtype=np.int32),
        "row_in_shard": np.asarray([record.row_in_shard for record in records], dtype=np.int32),
        "config_json": np.asarray([config_json], dtype=object),
        "surface_feature_names": np.asarray(surface_feature_names, dtype=object),
    }
    _save_npz(path, **arrays)


def write_content_shard(path: Path, shard: ContentArrayShard) -> None:
    arrays = {
        "shard_id": np.asarray([shard.shard_id], dtype=np.int32),
        "event_ids": np.asarray(shard.event_ids, dtype=object),
        "input_ids": np.asarray(shard.input_ids, dtype=np.int32),
        "attention_mask": np.asarray(shard.attention_mask, dtype=np.uint8),
    }
    if shard.surface_features is not None:
        arrays["surface_features"] = np.asarray(shard.surface_features, dtype=np.float32)
    if shard.quality_flags is not None:
        arrays["quality_flags"] = np.asarray(shard.quality_flags, dtype=np.uint8)
    if shard.primary_label_ids is not None:
        arrays["primary_label_ids"] = np.asarray(shard.primary_label_ids, dtype=np.int16)
    _save_npz(path, **arrays)
