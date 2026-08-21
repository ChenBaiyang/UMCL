from __future__ import annotations

import argparse
from pathlib import Path

from config.resolver import resolve_encoder_config
from preprocess.preprocess_events import build_default_args as build_preprocess_event_args
from preprocess.preprocess_events import preprocess_events
from utils.logging_utils import get_logger
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME
from utils.torch_runtime import resolve_torch_device


LOGGER = get_logger("preprocess_views")
ALL_VIEWS = ("content", "stance", "propagation", "evidence", "source")


def _ensure_preprocessed_events(dataset_id: str, input_root: Path) -> Path:
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if input_path.exists():
        return input_root

    LOGGER.info(
        "missing preprocessed events for %s at %s; running preprocess_events with default arguments",
        dataset_id,
        input_path,
    )
    preprocess_events(build_preprocess_event_args(dataset_id))
    default_input_root = Path("data/preprocessed")
    return default_input_root


def parse_views(raw: str | None) -> list[str]:
    if raw is None or raw.strip().lower() == "all":
        return list(ALL_VIEWS)
    views = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(views) - set(ALL_VIEWS))
    if invalid:
        raise ValueError(f"Unsupported views: {invalid}. Expected a subset of {list(ALL_VIEWS)}")
    deduped: list[str] = []
    for view in views:
        if view not in deduped:
            deduped.append(view)
    return deduped


def preprocess_views(
    dataset_id: str,
    mode: str,
    *,
    input_root: Path,
    data_root: Path,
    views: list[str],
    stance_device: str,
    stance_tokenizer_parallelism: str,
    stance_torch_num_threads: int | None,
    stance_gc_interval_shards: int,
    stance_encode_batch_size_override: int | None,
) -> dict[str, str]:
    input_root = _ensure_preprocessed_events(dataset_id, input_root)
    built: dict[str, str] = {}
    if "content" in views:
        from models.encoders.content_view.pipelines.build_content_arrays import build_arrays as build_content_arrays
        from models.encoders.content_view.pipelines.build_content_index import build_index as build_content_index

        content_config = resolve_encoder_config("content", dataset_id, mode)
        LOGGER.info("building content-view index for %s (%s)", dataset_id, mode)
        build_content_index(dataset_id, mode, input_root, data_root / "content_view", config=content_config)
        LOGGER.info("building content-view arrays for %s (%s)", dataset_id, mode)
        build_content_arrays(dataset_id, mode, input_root, data_root / "content_view", config=content_config)
        built["content"] = "built"

    if "stance" in views:
        stance_config = resolve_encoder_config("stance", dataset_id, mode)
        from models.encoders.stance_view.pipelines.build_stance_arrays import build_stance_arrays

        LOGGER.info("building stance-view arrays for %s (%s)", dataset_id, mode)
        resolved_device, fallback_reason = resolve_torch_device(stance_device)
        if fallback_reason is None:
            LOGGER.info("stance-view device resolved to %s (requested=%s)", resolved_device, stance_device)
        else:
            LOGGER.info(
                "stance-view device fallback to %s (requested=%s, reason=%s)",
                resolved_device,
                stance_device,
                fallback_reason,
            )
        build_stance_arrays(
            dataset_id,
            mode,
            input_root,
            data_root / "stance_view",
            device=stance_device,
            config=stance_config,
            tokenizer_parallelism=stance_tokenizer_parallelism,
            torch_num_threads=stance_torch_num_threads,
            gc_interval_shards=stance_gc_interval_shards,
            encode_batch_size_override=stance_encode_batch_size_override,
        )
        built["stance"] = "built"

    if "propagation" in views:
        propagation_config = resolve_encoder_config("propagation", dataset_id, mode)
        from models.encoders.propagation_view.pipelines.build_propagation_arrays import build_propagation_arrays

        LOGGER.info("building propagation-view arrays for %s (%s)", dataset_id, mode)
        build_propagation_arrays(
            dataset_id,
            mode,
            input_root,
            data_root / "propagation_view",
            config=propagation_config,
        )
        built["propagation"] = "built"

    if "evidence" in views:
        from models.encoders.evidence_view.pipelines.build_evidence_arrays import build_evidence_arrays

        evidence_config = resolve_encoder_config("evidence", dataset_id, mode)
        LOGGER.info("building evidence-view arrays for %s (%s)", dataset_id, mode)
        build_evidence_arrays(dataset_id, mode, input_root, data_root / "evidence_view", config=evidence_config)
        built["evidence"] = "built"

    if "source" in views:
        from models.encoders.source_view.pipelines.build_source_arrays import build_source_arrays

        source_config = resolve_encoder_config("source", dataset_id, mode)
        LOGGER.info("building source-view arrays for %s (%s)", dataset_id, mode)
        build_source_arrays(dataset_id, mode, input_root, data_root / "source_view", config=source_config)
        built["source"] = "built"
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build five-view artifacts from preprocessed events.")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--views", default="all", help="Comma-separated subset of content,stance,propagation,evidence,source")
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--stance_device", default="auto")
    parser.add_argument("--stance_tokenizer_parallelism", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--stance_torch_num_threads", type=int, default=None)
    parser.add_argument("--stance_gc_interval_shards", type=int, default=8)
    parser.add_argument("--stance_encode_batch_size_override", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_views(
        args.dataset_id,
        "full",
        input_root=Path(args.input_root),
        data_root=Path(args.data_root),
        views=parse_views(args.views),
        stance_device=args.stance_device,
        stance_tokenizer_parallelism=args.stance_tokenizer_parallelism,
        stance_torch_num_threads=args.stance_torch_num_threads,
        stance_gc_interval_shards=args.stance_gc_interval_shards,
        stance_encode_batch_size_override=args.stance_encode_batch_size_override,
    )


if __name__ == "__main__":
    main()
