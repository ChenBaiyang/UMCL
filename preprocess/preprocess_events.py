from __future__ import annotations

import argparse
import gc
import gzip
import os
import pickle
import time
from collections import Counter
from pathlib import Path

from config.resolver import resolve_event_pipeline_config
from preprocess.event_pipeline.cli_build import (
    GZIP_COMPRESSLEVEL,
    PROGRESS_LOG_EVERY,
    _build_split_assignments,
    _build_split_label_map,
    _write_split_manifest,
)
from preprocess.event_pipeline.dataset_ingest import available_dataset_ids, open_raw_dataset
from preprocess.event_pipeline.event_builder import UnifiedEventBuilder
from preprocess.event_pipeline.id_registry import IdCompactor
from preprocess.event_pipeline.preprocessing import preprocess_unified_event
from preprocess.event_pipeline.validation import validate_unified_event
from utils.jsonl import dataclass_to_dict, write_json
from utils.logging_utils import get_logger
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME


LOGGER = get_logger("preprocess_events")


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


def build_default_args(dataset_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_id=dataset_id,
        downloads_root="data/downloads",
        output_root="data/preprocessed",
        id_map_mode="summary",
        weibo_converter_workers=None,
        progress_log_every=PROGRESS_LOG_EVERY,
    )


def preprocess_events(args: argparse.Namespace) -> None:
    dataset_id = args.dataset_id
    output_dir = Path(args.output_root) / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / PREPROCESSED_EVENTS_FILENAME
    report_path = output_dir / "preprocessing_report.json"
    id_map_path = output_dir / "id_maps.json"
    full_id_map_path = output_dir / "id_maps.full.pkl.gz"

    config = resolve_event_pipeline_config(dataset_id=dataset_id, mode="full")
    pre_validation_warnings_total = 0
    pre_validation_errors_total = 0
    validation_warnings: list[str] = []
    validation_errors: list[str] = []
    split_counts: Counter[str] = Counter()
    event_count = 0
    platform_count: dict[str, int] = {}
    view_availability_count = {name: 0 for name in ["content", "stance", "propagation", "evidence", "source"]}
    placeholder_node_count = 0
    text_unobserved_node_count = 0
    boilerplate_node_count = 0
    relative_delay_only_node_count = 0
    source_only_evidence_count = 0
    missing_timestamp_count = 0
    missing_source_metadata_count = 0

    dataset = None
    builder = None
    compactor = None
    split_manifest_path: Path | None = None
    start_time = time.perf_counter()
    raw_open_end_time = start_time
    processing_end_time = start_time
    try:
        if args.weibo_converter_workers is not None:
            os.environ["UMCM_WEIBO_CONVERTER_WORKERS"] = str(max(1, args.weibo_converter_workers))
        LOGGER.info(
            "starting raw dataset open for %s (weibo_converter_workers=%s)",
            dataset_id,
            os.environ.get("UMCM_WEIBO_CONVERTER_WORKERS", "auto"),
        )
        with open_raw_dataset(dataset_id, args.downloads_root) as dataset:
            raw_open_end_time = time.perf_counter()
            LOGGER.info("loaded raw dataset for %s in %.2fs", dataset_id, raw_open_end_time - start_time)
            builder = UnifiedEventBuilder(dataset)
            compactor = IdCompactor(dataset_id)
            annotation_rows = list(dataset.iter_annotations())
            compactor.prime_annotation_ids(annotation_rows)
            event_rows = list(dataset.iter_events())
            label_map = _build_split_label_map(dataset_id, annotation_rows)
            split_assignments = _build_split_assignments(
                dataset_id,
                event_rows,
                annotation_rows,
                config.split_seed,
                label_map=label_map,
            )
            split_manifest_path = _write_split_manifest(
                dataset_id=dataset_id,
                split_seed=config.split_seed,
                downloads_root=args.downloads_root,
                event_rows=event_rows,
                split_assignments=split_assignments,
                label_map=label_map,
                dataset=dataset,
            )
            LOGGER.info("wrote split manifest for %s to %s", dataset_id, split_manifest_path)

            with gzip.open(output_path, mode="wb", compresslevel=GZIP_COMPRESSLEVEL) as handle:
                for event_row in event_rows:
                    unified = builder.build_from_event_row(event_row)
                    assignment = split_assignments[str(event_row["event_id"])]
                    unified.canonical_split = assignment.canonical_split
                    unified.raw_split_source = assignment.raw_split_source
                    pre_warnings, pre_errors = validate_unified_event(unified)
                    pre_validation_warnings_total += len(pre_warnings)
                    pre_validation_errors_total += len(pre_errors)

                    event = preprocess_unified_event(unified, config)
                    warnings, errors = validate_unified_event(event)
                    if len(validation_warnings) < 100:
                        validation_warnings.extend(warnings[: max(0, 100 - len(validation_warnings))])
                    if len(validation_errors) < 100:
                        validation_errors.extend(errors[: max(0, 100 - len(validation_errors))])

                    payload = dataclass_to_dict(event)
                    payload = compactor.compact_event_row(payload)
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    event_count += 1
                    split_counts[event.canonical_split] += 1
                    platform_count[event.platform] = platform_count.get(event.platform, 0) + 1
                    for name, observed in event.observed_view_mask.items():
                        view_availability_count[name] += int(observed)
                    if event.propagation:
                        placeholder_node_count += event.propagation.summary.n_placeholders
                        text_unobserved_node_count += event.propagation.summary.n_text_unobserved_nodes
                        boilerplate_node_count += event.propagation.summary.n_boilerplate_nodes
                        relative_delay_only_node_count += sum(
                            1 for node in event.propagation.nodes if node.time.observation_kind == "relative_delay_observed"
                        )
                        missing_timestamp_count += sum(
                            1 for node in event.propagation.nodes if node.time.observation_kind == "missing"
                        )
                    if event.evidence and event.evidence.summary.has_explicit_ref:
                        source_only_evidence_count += 1
                    if event.source:
                        missing_source_metadata_count += sum(
                            1
                            for field_name in [
                                "account_created_at",
                                "verified",
                                "followers_count",
                                "following_count",
                                "post_count",
                                "profile_description_raw",
                                "display_name_raw",
                            ]
                            if getattr(event.source.profile.missingness, field_name) == "missing"
                        )

                    if event_count % max(1, args.progress_log_every) == 0:
                        LOGGER.info(
                            "processed %s events for %s (platform_breakdown=%s)",
                            event_count,
                            dataset_id,
                            platform_count,
                        )

                    del payload, event, unified, pre_warnings, pre_errors, warnings, errors
            processing_end_time = time.perf_counter()
            LOGGER.info(
                "finished event build/preprocess loop for %s in %.2fs",
                dataset_id,
                processing_end_time - raw_open_end_time,
            )
    finally:
        if builder is not None:
            del builder
        if dataset is not None:
            try:
                dataset.close()
            except AttributeError:
                pass
        gc.collect()

    if args.id_map_mode == "summary":
        write_json(
            id_map_path,
            compactor.export_summary() if compactor is not None else {"dataset_id": dataset_id, "counts": {}, "samples": {}},
        )
    elif args.id_map_mode == "full":
        with gzip.open(full_id_map_path, mode="wb", compresslevel=GZIP_COMPRESSLEVEL) as handle:
            pickle.dump(
                compactor.export_compact_payload()
                if compactor is not None
                else {"dataset_id": dataset_id, "format": "compact_id_map_v1", "id_spaces": {}, "counts": {}},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    write_json(
        report_path,
        {
            "dataset_id": dataset_id,
            "event_count": event_count,
            "platform_count": platform_count,
            "view_availability_count": view_availability_count,
            "placeholder_node_count": placeholder_node_count,
            "text_unobserved_node_count": text_unobserved_node_count,
            "boilerplate_node_count": boilerplate_node_count,
            "relative_delay_only_node_count": relative_delay_only_node_count,
            "source_only_evidence_count": source_only_evidence_count,
            "missing_timestamp_count": missing_timestamp_count,
            "missing_source_metadata_count": missing_source_metadata_count,
            "pre_validation_warning_count": pre_validation_warnings_total,
            "pre_validation_error_count": pre_validation_errors_total,
            "validation_warnings": validation_warnings,
            "validation_errors": validation_errors,
            "split_counts": dict(split_counts),
            "split_seed": config.split_seed,
            "split_manifest_path": str(split_manifest_path) if split_manifest_path is not None else None,
            "id_map_path": (
                str(id_map_path)
                if args.id_map_mode == "summary"
                else str(full_id_map_path)
                if args.id_map_mode == "full"
                else None
            ),
            "id_map_mode": args.id_map_mode,
            "mode": "full",
            "resolved_pipeline_config": config.as_dict(),
            "output": str(output_path),
            "output_format": "pickle+gzip",
            "timing": {
                "raw_open_sec": raw_open_end_time - start_time,
                "build_preprocess_sec": processing_end_time - raw_open_end_time,
                "total_sec": time.perf_counter() - start_time,
            },
        },
    )
    if compactor is not None:
        del compactor
    LOGGER.info(
        "built %s preprocessed events directly from raw downloads for %s in %.2fs",
        event_count,
        dataset_id,
        time.perf_counter() - start_time,
    )


def main() -> None:
    preprocess_events(parse_args())


if __name__ == "__main__":
    main()
