from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from models.encoders.content_view.artifacts import EVENT_INDEX_FILENAME
from config.resolver import resolve_encoder_config
from models.encoders.content_view.array_writer import write_content_shard
from models.encoders.content_view.config import ContentViewConfig
from models.encoders.content_view.dataset import iter_preprocessed_events, load_event_index
from models.encoders.content_view.feature_builder import build_surface_feature_vector
from models.encoders.content_view.label_alignment import align_content_labels
from models.encoders.content_view.schemas import ContentArrayShard
from models.encoders.content_view.text_builder import build_encoder_text, encoder_text_lengths
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME
from models.encoders.content_view.tokenizer_builder import build_tokenizer, tokenize_texts


def build_arrays(dataset_id: str, mode: str, input_root: Path, output_root: Path, *, config: ContentViewConfig | None = None) -> None:
    config = config or resolve_encoder_config("content", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    data_root = input_root.parent
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    index_path = output_root / dataset_id / artifact_mode / EVENT_INDEX_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing content index file: {index_path}")

    index_payload = load_event_index(index_path)
    expected_event_ids = list(index_payload["event_ids"])
    expected_primary_label_ids = index_payload["primary_label_ids"]
    expected_shard_ids = index_payload["shard_ids"]
    expected_rows = index_payload["row_in_shard"]

    tokenizer = build_tokenizer(config, mode)
    shard_id = None
    shard_events: list[str] = []
    shard_texts: list[str] = []
    shard_surface: list[np.ndarray] = []
    shard_quality: list[list[int]] = []
    shard_primary_labels: list[int] = []

    def flush(current_shard_id: int | None) -> None:
        if current_shard_id is None or not shard_events:
            return
        input_ids, attention_mask = tokenize_texts(tokenizer, shard_texts, config, mode)
        shard = ContentArrayShard(
            shard_id=current_shard_id,
            event_ids=np.asarray(shard_events, dtype=object),
            input_ids=np.asarray(input_ids, dtype=np.int32),
            attention_mask=np.asarray(attention_mask, dtype=np.uint8),
            surface_features=np.asarray(shard_surface, dtype=np.float32),
            quality_flags=np.asarray(shard_quality, dtype=np.uint8),
            primary_label_ids=np.asarray(shard_primary_labels, dtype=np.int16),
        )
        write_content_shard(output_root / dataset_id / artifact_mode / f"content_shard_{artifact_mode}_{current_shard_id:04d}.npz", shard)

    for row_index, event_row in enumerate(iter_preprocessed_events(input_path)):
        if row_index >= len(expected_event_ids):
            raise ValueError("Preprocessed events contain more rows than content index.")

        event_id = event_row["event_id"]
        expected_event_id = expected_event_ids[row_index]
        if event_id != expected_event_id:
            raise ValueError(f"Index/order mismatch at row {row_index}: {event_id} != {expected_event_id}")

        encoder_text, quality = build_encoder_text(event_row, config, mode)
        _, tokens_approx = encoder_text_lengths(encoder_text)
        features = build_surface_feature_vector(event_row, encoder_text, config, tokens_approx)
        labels = align_content_labels(event_row["dataset_id"], event_row.get("task_targets") or {}, data_root=str(data_root))
        if int(labels["primary_label_id"]) != int(expected_primary_label_ids[row_index]):
            raise ValueError(f"Primary label mismatch for event {event_id}")

        target_shard_id = int(expected_shard_ids[row_index])
        expected_row = int(expected_rows[row_index])
        if shard_id is None:
            shard_id = target_shard_id
        if target_shard_id != shard_id:
            flush(shard_id)
            shard_events.clear()
            shard_texts.clear()
            shard_surface.clear()
            shard_quality.clear()
            shard_primary_labels.clear()
            shard_id = target_shard_id
        if expected_row != len(shard_events):
            raise ValueError(f"Shard row mismatch for event {event_id}: expected {expected_row}, got {len(shard_events)}")

        shard_events.append(event_id)
        shard_texts.append(encoder_text)
        shard_surface.append(features)
        shard_quality.append(
            [
                int(quality["is_text_missing"]),
                int(quality["is_text_empty"]),
                int(quality["is_text_boilerplate"]),
                int(quality["is_text_substantive"]),
                int(quality["is_semantically_usable"]),
            ]
        )
        shard_primary_labels.append(int(labels["primary_label_id"]))

    flush(shard_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/content_view")
    args = parser.parse_args()
    config = resolve_encoder_config("content", args.dataset_id, "full")
    build_arrays(args.dataset_id, "full", Path(args.input_root), Path(args.output_root), config=config)


if __name__ == "__main__":
    main()
