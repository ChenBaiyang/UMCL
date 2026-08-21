from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config.resolver import resolve_encoder_config
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.encoders.source_view.array_writer import write_source_event_meta, write_source_manifest, write_source_shard
from models.encoders.source_view.config import Mode, SourceViewConfig
from models.encoders.source_view.dataset import iter_preprocessed_events
from models.encoders.source_view.label_alignment import MISSING_LABEL_ID, align_source_event_labels
from models.encoders.source_view.preprocessing import apply_summary_normalization, build_summary_normalization_stats
from models.encoders.source_view.schemas import SourceArrayShard, SourceEventMeta
from models.encoders.source_view.summary_builder import build_source_summary
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME


def build_source_arrays(
    dataset_id: str,
    mode: Mode,
    input_root: Path,
    output_root: Path,
    *,
    config: SourceViewConfig | None = None,
) -> None:
    config = config or resolve_encoder_config("source", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    data_root = input_root.parent
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")
    mode_dir = output_root / dataset_id / artifact_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    event_meta_rows: list[dict] = []
    shard_filenames: list[str] = []
    shard_id = 0
    source_mode_counts: dict[str, int] = {}
    all_event_ids: list[str] = []
    all_summary_features: list[np.ndarray] = []
    all_quality_flags: list[np.ndarray] = []
    all_categorical_features: list[np.ndarray] = []
    all_primary_label_ids: list[int] = []
    all_canonical_splits: list[str] = []
    primary_class_names: list[str] = []

    def flush_shard(
        shard_event_ids: list[str],
        shard_summary_features: list[np.ndarray],
        shard_quality_flags: list[np.ndarray],
        shard_categorical_features: list[np.ndarray],
        shard_primary_label_ids: list[int],
    ) -> None:
        nonlocal shard_id
        if not shard_event_ids:
            return
        filename = f"source_shard_{artifact_mode}_{shard_id:04d}.npz"
        shard = SourceArrayShard(
            shard_id=shard_id,
            event_ids=np.asarray(shard_event_ids, dtype=object),
            summary_features=np.asarray(shard_summary_features, dtype=np.float32),
            summary_quality_flags=np.asarray(shard_quality_flags, dtype=np.uint8),
            categorical_features=np.asarray(shard_categorical_features, dtype=np.int16),
            primary_label_ids=np.asarray(shard_primary_label_ids, dtype=np.int32),
            n_events=len(shard_event_ids),
        )
        write_source_shard(mode_dir / filename, shard)
        shard_filenames.append(filename)
        shard_event_ids.clear()
        shard_summary_features.clear()
        shard_quality_flags.clear()
        shard_categorical_features.clear()
        shard_primary_label_ids.clear()
        shard_id += 1

    for event_row in iter_preprocessed_events(input_path):
        labels = align_source_event_labels(dataset_id, event_row.get("task_targets") or {}, data_root=str(data_root))
        label_id = int(labels["primary_label_id"]) if int(labels["primary_label_id"]) != MISSING_LABEL_ID else -1
        primary_class_names = list(labels["primary_class_names"])

        payload, summary_vector, quality_vector, categorical_vector = build_source_summary(config, event_row)
        source_mode = payload["source_mode"]
        source_mode_counts[source_mode] = source_mode_counts.get(source_mode, 0) + 1
        profile = ((event_row.get("source") or {}).get("profile")) or {}
        platform_extra = profile.get("platform_extra") or {}

        event_meta = SourceEventMeta(
            event_id=event_row["event_id"],
            dataset_id=dataset_id,
            platform=event_row["platform"],
            canonical_split=event_row["canonical_split"],
            raw_split_source=event_row.get("raw_split_source"),
            primary_task_name=str(labels["primary_task_name"]),
            primary_metric_name=str(labels["primary_metric_name"]),
            primary_label_name=labels["primary_label_name"],
            primary_label_id=label_id,
            source_user_id=(event_row.get("source") or {}).get("source_user_id"),
            source_mode=source_mode,
            source_signal_observed=bool(payload["quality"]["source_signal_observed"]),
            source_sparse_mask=bool(payload["quality"]["source_sparse_mask"]),
            source_identity_sparse_mask=bool(payload["quality"]["source_identity_sparse_mask"]),
            source_social_sparse_mask=bool(payload["quality"]["source_social_sparse_mask"]),
            source_role_sparse_mask=bool(payload["quality"]["source_role_sparse_mask"]),
            source_history_sparse_mask=bool(payload["quality"]["source_history_sparse_mask"]),
            verified_flag=float(payload["summary"]["verified_flag"]),
            has_account_created_at=float(payload["summary"]["has_account_created_at"]),
            account_age_days=float(payload["summary"]["account_age_days"]),
            has_profile_description=float(payload["summary"]["has_profile_description"]),
            has_display_name=float(payload["summary"]["has_display_name"]),
            profile_description_length_chars=float(payload["summary"]["profile_description_length_chars"]),
            display_name_length_chars=float(payload["summary"]["display_name_length_chars"]),
            profile_completeness_count=float(payload["summary"]["profile_completeness_count"]),
            avatar_present_flag=float(payload["summary"]["avatar_present_flag"]),
            location_present_flag=float(payload["summary"]["location_present_flag"]),
            gender_present_flag=float(payload["summary"]["gender_present_flag"]),
            verified_reason_present_flag=float(payload["summary"]["verified_reason_present_flag"]),
            verified_type_present_flag=float(payload["summary"]["verified_type_present_flag"]),
            followers_count_raw=float(payload["summary"]["followers_count_raw"]),
            following_count_raw=float(payload["summary"]["following_count_raw"]),
            follower_followee_ratio=float(payload["summary"]["follower_followee_ratio"]),
            post_count_raw=float(payload["summary"]["post_count_raw"]),
            listed_count_raw=float(payload["summary"]["listed_count_raw"]),
            membership_indicator_bi_followers_raw=float(payload["summary"]["membership_indicator_bi_followers_raw"]),
            favourites_count_raw=float(payload["summary"]["favourites_count_raw"]),
            geo_enabled_flag=float(payload["summary"]["geo_enabled_flag"]),
            lifetime_posting_rate_proxy=float(payload["summary"]["lifetime_posting_rate_proxy"]),
            explicit_role_present=float(payload["summary"]["explicit_role_present"]),
            verified_reason_present=float(payload["summary"]["verified_reason_present"]),
            verified_type_id=int(payload["categorical"]["verified_type_id"]),
            role_bucket_id=int(payload["categorical"]["role_bucket_id"]),
            age_bucket_id=int(payload["categorical"]["age_bucket_id"]),
            activity_rate_bucket_id=int(payload["categorical"]["activity_rate_bucket_id"]),
            summary_feature_dim=int(summary_vector.shape[0]),
            notes=f"platform_extra_keys={sorted(platform_extra.keys())[:6]}",
        )
        event_meta_rows.append(event_meta.__dict__.copy())
        all_event_ids.append(event_row["event_id"])
        all_summary_features.append(summary_vector)
        all_quality_flags.append(quality_vector)
        all_categorical_features.append(categorical_vector)
        all_primary_label_ids.append(label_id if label_id != MISSING_LABEL_ID else -1)
        all_canonical_splits.append(str(event_row["canonical_split"]))

    normalization_stats: dict[str, dict[str, float | int | str]] = {}
    normalized_summary_features = np.asarray(all_summary_features, dtype=np.float32)
    quality_flags_array = np.asarray(all_quality_flags, dtype=np.uint8)
    if config.normalize_summary_features and normalized_summary_features.size > 0:
        normalization_stats = build_summary_normalization_stats(
            config,
            normalized_summary_features,
            quality_flags_array,
            all_canonical_splits,
        )
        normalized_summary_features = apply_summary_normalization(
            config,
            normalized_summary_features,
            quality_flags_array,
            normalization_stats,
        )

    shard_event_ids: list[str] = []
    shard_summary_features: list[np.ndarray] = []
    shard_quality_flags: list[np.ndarray] = []
    shard_categorical_features: list[np.ndarray] = []
    shard_primary_label_ids: list[int] = []
    for idx, event_id in enumerate(all_event_ids):
        shard_event_ids.append(event_id)
        shard_summary_features.append(normalized_summary_features[idx])
        shard_quality_flags.append(quality_flags_array[idx])
        shard_categorical_features.append(all_categorical_features[idx])
        shard_primary_label_ids.append(all_primary_label_ids[idx])
        if len(shard_event_ids) >= config.shard_size:
            flush_shard(
                shard_event_ids,
                shard_summary_features,
                shard_quality_flags,
                shard_categorical_features,
                shard_primary_label_ids,
            )
    flush_shard(
        shard_event_ids,
        shard_summary_features,
        shard_quality_flags,
        shard_categorical_features,
        shard_primary_label_ids,
    )
    write_source_event_meta(mode_dir / "source_event_meta.jsonl", event_meta_rows)
    manifest = {
        "dataset_id": dataset_id,
        "mode": artifact_mode,
        "config_snapshot": config.as_dict(),
        "event_count": len(event_meta_rows),
        "shard_count": len(shard_filenames),
        "shard_filenames": shard_filenames,
        "summary_feature_names": config.summary_feature_names,
        "summary_quality_flag_names": config.summary_quality_flag_names,
        "categorical_feature_names": config.categorical_feature_names,
        "primary_task_name": event_meta_rows[0]["primary_task_name"] if event_meta_rows else None,
        "primary_metric_name": event_meta_rows[0]["primary_metric_name"] if event_meta_rows else None,
        "primary_class_names": primary_class_names if event_meta_rows else [],
        "missing_label_id": -1,
        "source_mode_counts": source_mode_counts,
        "summary_normalization": {
            "enabled": bool(config.normalize_summary_features),
            "normalized_feature_names": list(normalization_stats.keys()),
            "feature_stats": normalization_stats,
        },
    }
    write_source_manifest(mode_dir / f"source_manifest_{artifact_mode}.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/source_view")
    args = parser.parse_args()
    config = resolve_encoder_config("source", args.dataset_id, "full")
    build_source_arrays(args.dataset_id, "full", Path(args.input_root), Path(args.output_root), config=config)


if __name__ == "__main__":
    main()
