from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config.resolver import resolve_encoder_config
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.encoders.evidence_view.array_writer import write_evidence_event_meta, write_evidence_manifest, write_evidence_reference_meta, write_evidence_shard
from models.encoders.evidence_view.config import EvidenceViewConfig, Mode
from models.encoders.evidence_view.dataset import iter_preprocessed_events
from models.encoders.evidence_view.label_alignment import MISSING_LABEL_ID, align_evidence_event_labels
from models.encoders.evidence_view.reference_builder import build_reference_rows
from models.encoders.evidence_view.schemas import EvidenceArrayShard, EvidenceEventMeta
from models.encoders.evidence_view.summary_builder import build_summary_features
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME


def build_evidence_arrays(
    dataset_id: str,
    mode: Mode,
    input_root: Path,
    output_root: Path,
    *,
    config: EvidenceViewConfig | None = None,
) -> None:
    config = config or resolve_encoder_config("evidence", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    data_root = input_root.parent
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")
    mode_dir = output_root / dataset_id / artifact_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    event_meta_rows: list[dict] = []
    ref_meta_rows: list[dict] = []
    shard_filenames: list[str] = []
    shard_id = 0

    shard_event_ids: list[str] = []
    shard_summary_features: list[np.ndarray] = []
    shard_quality_flags: list[np.ndarray] = []
    shard_primary_label_ids: list[int] = []
    shard_ref_features: list[np.ndarray] = []
    shard_ref_event_index: list[int] = []
    shard_ref_mask: list[int] = []

    total_ref_rows = 0
    reference_event_count = 0
    evidence_mode_counts: dict[str, int] = {}

    def flush_shard() -> None:
        nonlocal shard_id
        if not shard_event_ids:
            return
        filename = f"evidence_shard_{artifact_mode}_{shard_id:04d}.npz"
        ref_row_features = np.asarray(np.vstack(shard_ref_features), dtype=np.float32) if shard_ref_features else np.zeros((0, 8), dtype=np.float32)
        ref_dim = int(ref_row_features.shape[1]) if ref_row_features.ndim == 2 and ref_row_features.shape[0] > 0 else 8
        max_ref_rows = max((features.shape[0] for features in shard_ref_features), default=0)
        event_ref_row_features = np.zeros((len(shard_event_ids), max_ref_rows, ref_dim), dtype=np.float32)
        event_ref_row_mask = np.zeros((len(shard_event_ids), max_ref_rows), dtype=np.uint8)
        for event_idx, event_ref_features in enumerate(shard_ref_features):
            n_rows = int(event_ref_features.shape[0])
            if n_rows == 0:
                continue
            event_ref_row_features[event_idx, :n_rows, :] = event_ref_features
            event_ref_row_mask[event_idx, :n_rows] = 1
        shard = EvidenceArrayShard(
            shard_id=shard_id,
            event_ids=np.asarray(shard_event_ids, dtype=object),
            summary_features=np.asarray(shard_summary_features, dtype=np.float32),
            summary_quality_flags=np.asarray(shard_quality_flags, dtype=np.uint8),
            primary_label_ids=np.asarray(shard_primary_label_ids, dtype=np.int32),
            ref_row_features=ref_row_features,
            ref_row_event_index=np.asarray(shard_ref_event_index, dtype=np.int32),
            ref_row_mask=np.asarray(shard_ref_mask, dtype=np.uint8),
            event_ref_row_features=event_ref_row_features,
            event_ref_row_mask=event_ref_row_mask,
            n_events=len(shard_event_ids),
            n_ref_rows=int(ref_row_features.shape[0]),
        )
        write_evidence_shard(mode_dir / filename, shard)
        shard_filenames.append(filename)
        shard_event_ids.clear()
        shard_summary_features.clear()
        shard_quality_flags.clear()
        shard_primary_label_ids.clear()
        shard_ref_features.clear()
        shard_ref_event_index.clear()
        shard_ref_mask.clear()
        shard_id += 1

    for event_row in iter_preprocessed_events(input_path):
        labels = align_evidence_event_labels(dataset_id, event_row.get("task_targets") or {}, data_root=str(data_root))
        label_id = int(labels["primary_label_id"]) if int(labels["primary_label_id"]) != MISSING_LABEL_ID else -1

        payload, summary_vector, quality_vector = build_summary_features(config, event_row)
        evidence = payload["summary"]
        quality = payload["quality"]
        refs = ((event_row.get("evidence") or {}).get("references")) or []
        ref_metas, ref_features = build_reference_rows(config, dataset_id, event_row["platform"], event_row["event_id"], refs)

        if not quality["evidence_signal_observed"]:
            evidence_mode = "no_signal"
        elif quality["sparse_evidence_mask"]:
            evidence_mode = "sparse"
        else:
            evidence_mode = "observed"
        evidence_mode_counts[evidence_mode] = evidence_mode_counts.get(evidence_mode, 0) + 1
        if refs:
            reference_event_count += 1

        event_meta = EvidenceEventMeta(
            event_id=event_row["event_id"],
            dataset_id=dataset_id,
            platform=event_row["platform"],
            canonical_split=event_row["canonical_split"],
            raw_split_source=event_row.get("raw_split_source"),
            primary_task_name=str(labels["primary_task_name"]),
            primary_metric_name=str(labels["primary_metric_name"]),
            primary_label_name=labels["primary_label_name"],
            primary_label_id=label_id,
            source_post_id=((event_row.get("evidence") or {}).get("source_post_id")) or "",
            evidence_mode=evidence_mode,
            has_explicit_ref=bool(evidence["has_explicit_ref"]),
            has_any_source_evidence_signal=bool(evidence["has_any_source_evidence_signal"]),
            n_refs_total=int(evidence["n_refs_total"]),
            n_url_refs=int(evidence["n_url_refs"]),
            n_media_refs=int(evidence["n_media_refs"]),
            n_internal_refs=int(evidence["n_internal_refs"]),
            n_external_refs=int(evidence["n_external_refs"]),
            n_reference_types_present=int(evidence["n_reference_types_present"]),
            has_multi_ref_type=bool(evidence["has_multi_ref_type"]),
            has_multi_ref_rows=bool(evidence["has_multi_ref_rows"]),
            has_specific_reference_signal=bool(evidence["has_specific_reference_signal"]),
            evidence_signal_observed=bool(quality["evidence_signal_observed"]),
            no_reference_rows_mask=bool(quality["no_reference_rows_mask"]),
            weak_evidence_mask=bool(quality["weak_evidence_mask"]),
            sparse_evidence_mask=bool(quality["sparse_evidence_mask"]),
            authority_unknown_mask=bool(quality["authority_unknown_mask"]),
            summary_feature_dim=int(summary_vector.shape[0]),
            notes="",
        )
        event_meta_rows.append(event_meta.__dict__.copy())
        for ref_meta in ref_metas:
            ref_meta_rows.append(ref_meta.__dict__.copy())

        shard_event_row = len(shard_event_ids)
        shard_event_ids.append(event_row["event_id"])
        shard_summary_features.append(summary_vector)
        shard_quality_flags.append(quality_vector)
        shard_primary_label_ids.append(label_id if label_id != MISSING_LABEL_ID else -1)
        shard_ref_features.append(ref_features)
        if ref_features.shape[0] > 0:
            shard_ref_event_index.extend([shard_event_row] * ref_features.shape[0])
            shard_ref_mask.extend([1] * ref_features.shape[0])
            total_ref_rows += int(ref_features.shape[0])
        if len(shard_event_ids) >= config.shard_size:
            flush_shard()

    flush_shard()
    write_evidence_event_meta(mode_dir / "evidence_event_meta.jsonl", event_meta_rows)
    if ref_meta_rows:
        write_evidence_reference_meta(mode_dir / "evidence_reference_meta.jsonl", ref_meta_rows)
    manifest = {
        "dataset_id": dataset_id,
        "mode": artifact_mode,
        "config_snapshot": config.as_dict(),
        "event_count": len(event_meta_rows),
        "reference_event_count": reference_event_count,
        "total_reference_row_count": total_ref_rows,
        "shard_count": len(shard_filenames),
        "shard_filenames": shard_filenames,
        "summary_feature_names": config.summary_feature_names,
        "summary_quality_flag_names": config.summary_quality_flag_names,
        "ref_row_feature_names": [
            "ref_type_id",
            "authority_bucket_id",
            "has_normalized_value",
            "has_domain",
            "has_target_post_id",
            "is_internal_platform_ref",
            "is_external_ref",
            "source_channel_id",
        ],
        "primary_task_name": event_meta_rows[0]["primary_task_name"] if event_meta_rows else None,
        "primary_metric_name": event_meta_rows[0]["primary_metric_name"] if event_meta_rows else None,
        "primary_class_names": list(labels["primary_class_names"]) if event_meta_rows else [],
        "missing_label_id": -1,
        "evidence_mode_counts": evidence_mode_counts,
        "uses_reference_rows": True,
    }
    write_evidence_manifest(mode_dir / f"evidence_manifest_{artifact_mode}.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/evidence_view")
    args = parser.parse_args()
    config = resolve_encoder_config("evidence", args.dataset_id, "full")
    build_evidence_arrays(args.dataset_id, "full", Path(args.input_root), Path(args.output_root), config=config)


if __name__ == "__main__":
    main()
