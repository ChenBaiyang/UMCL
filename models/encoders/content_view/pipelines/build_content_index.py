from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.encoders.content_view.artifacts import EVENT_INDEX_FILENAME
from config.resolver import resolve_encoder_config
from models.encoders.content_view.array_writer import write_content_index
from models.encoders.content_view.config import ContentViewConfig
from models.encoders.content_view.feature_builder import build_surface_feature_vector
from models.encoders.content_view.dataset import iter_preprocessed_events
from models.encoders.content_view.label_alignment import align_content_labels
from models.encoders.content_view.schemas import ContentIndexRecord
from models.encoders.content_view.text_builder import build_encoder_text, encoder_text_lengths
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME
def build_index(dataset_id: str, mode: str, input_root: Path, output_root: Path, *, config: ContentViewConfig | None = None) -> None:
    config = config or resolve_encoder_config("content", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    data_root = input_root.parent
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")

    records: list[ContentIndexRecord] = []
    seen_event_ids: set[str] = set()
    primary_class_names: list[str] = []

    for row_index, event_row in enumerate(iter_preprocessed_events(input_path)):
        event_id = event_row["event_id"]
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate event_id in content index build: {event_id}")
        seen_event_ids.add(event_id)

        encoder_text, quality = build_encoder_text(event_row, config, mode)
        chars, tokens_approx = encoder_text_lengths(encoder_text)
        features = build_surface_feature_vector(event_row, encoder_text, config, tokens_approx)
        labels = align_content_labels(event_row["dataset_id"], event_row.get("task_targets") or {}, data_root=str(data_root))
        if not primary_class_names:
            primary_class_names = list(labels["primary_class_names"])
        surface = ((event_row.get("content") or {}).get("surface")) or {}

        records.append(
            ContentIndexRecord(
                event_id=event_id,
                dataset_id=event_row["dataset_id"],
                platform=event_row["platform"],
                canonical_split=event_row.get("canonical_split") or "unknown",
                raw_split_source=event_row.get("raw_split_source"),
                primary_task_name=str(labels["primary_task_name"]),
                primary_metric_name=str(labels["primary_metric_name"]),
                primary_label_name=labels["primary_label_name"],
                primary_label_id=int(labels["primary_label_id"]),
                encoder_text_length_chars=chars,
                encoder_text_length_tokens_approx=tokens_approx,
                is_text_missing=quality["is_text_missing"],
                is_text_empty=quality["is_text_empty"],
                is_text_boilerplate=quality["is_text_boilerplate"],
                is_text_substantive=quality["is_text_substantive"],
                is_semantically_usable=quality["is_semantically_usable"],
                has_url=1 if bool(surface.get("has_url")) else 0,
                has_media=1 if bool(surface.get("has_media")) else 0,
                surface_feature_dim=int(features.shape[0]),
                shard_id=row_index // config.shard_size,
                row_in_shard=row_index % config.shard_size,
            )
        )

    output_dir = output_root / dataset_id / artifact_mode
    write_content_index(
        output_dir / EVENT_INDEX_FILENAME,
        records,
        config_json=json.dumps(config.as_dict(), ensure_ascii=False),
        surface_feature_names=config.surface_feature_names,
        primary_class_names=primary_class_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/content_view")
    args = parser.parse_args()
    config = resolve_encoder_config("content", args.dataset_id, "full")
    build_index(args.dataset_id, "full", Path(args.input_root), Path(args.output_root), config=config)


if __name__ == "__main__":
    main()
