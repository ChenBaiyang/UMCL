from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from config.resolver import resolve_encoder_config
from models.encoders.stance_view.array_writer import write_stance_label_map, write_stance_manifest, write_stance_shard
from models.encoders.stance_view.config import Mode, StanceViewConfig
from models.encoders.stance_view.dataset import iter_preprocessed_events
from models.encoders.stance_view.feature_builder import build_context_cat, build_context_dense
from models.encoders.stance_view.label_alignment import PRIMARY_LABEL_2WAY_NAMES, PRIMARY_LABEL_4WAY_NAMES, STANCE_LABEL_NAMES, align_event_labels, align_reply_label
from models.encoders.stance_view.relation_builder import build_source_aware_text, normalize_social_text
from models.encoders.stance_view.reply_selector import classify_reply_for_stance
from models.encoders.stance_view.sampling import sample_replies_deterministically
from models.encoders.stance_view.schemas import StanceArrayShard, StanceEventMeta
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME
from utils.torch_runtime import configure_torch_runtime, resolve_torch_device


LOG_EVERY_EVENTS = 100
DEFAULT_GC_INTERVAL_SHARDS = 8


def _encode_texts(texts: list[str], tokenizer, model, max_length: int, batch_size: int, device: Any) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    import torch

    if not texts:
        hidden_size = int(getattr(model.config, "hidden_size"))
        return np.zeros((0, hidden_size), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            del outputs, hidden, mask, encoded, pooled
    return np.concatenate(embeddings, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim == 2 else 0,), dtype=np.float32)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denominator = np.clip(denominator, a_min=1e-8, a_max=None)
    return (numerator / denominator).astype(np.float32, copy=False)


def _text_value(bundle: dict[str, Any] | None) -> str:
    bundle = bundle or {}
    return str(bundle.get("norm_light") or bundle.get("raw") or "")


def _write_shard(mode_dir: Path, artifact_mode: str, shard_id: int, rows: list[dict[str, Any]]) -> None:
    shard = StanceArrayShard(
        shard_id=shard_id,
        reply_embedding=np.asarray([row["reply_embedding"] for row in rows], dtype=np.float32),
        context_cat=np.asarray([row["context_cat"] for row in rows], dtype=np.int16),
        context_dense=np.asarray([row["context_dense"] for row in rows], dtype=np.float32),
        reply_label_ids=np.asarray([row["reply_label_id"] for row in rows], dtype=np.int16),
        reply_label_mask=np.asarray([row["reply_label_mask"] for row in rows], dtype=np.uint8),
        event_index=np.asarray([row["event_index"] for row in rows], dtype=np.int32),
        event_sampled_reply_count=np.asarray([row["event_sampled_reply_count"] for row in rows], dtype=np.int16),
        event_truncation_flag=np.asarray([row["event_truncation_flag"] for row in rows], dtype=np.uint8),
    )
    write_stance_shard(mode_dir / f"stance_shard_{artifact_mode}_{shard_id:04d}.npz", shard)


def _flush_pending_rows(
    pending_rows: list[dict[str, Any]],
    *,
    mode_dir: Path,
    artifact_mode: str,
    shard_id: int,
    gc_interval_shards: int,
) -> tuple[int, float]:
    if not pending_rows:
        return shard_id, 0.0
    write_started_at = time.perf_counter()
    _write_shard(mode_dir, artifact_mode, shard_id, pending_rows)
    write_elapsed = time.perf_counter() - write_started_at
    pending_rows.clear()
    next_shard_id = shard_id + 1
    if gc_interval_shards > 0 and next_shard_id % gc_interval_shards == 0:
        gc.collect()
    return next_shard_id, write_elapsed


def build_stance_arrays(
    dataset_id: str,
    mode: Mode,
    input_root: Path,
    output_root: Path,
    device: str = "auto",
    *,
    config: StanceViewConfig | None = None,
    tokenizer_parallelism: str = "auto",
    torch_num_threads: int | None = None,
    gc_interval_shards: int = DEFAULT_GC_INTERVAL_SHARDS,
    encode_batch_size_override: int | None = None,
) -> None:
    config = config or resolve_encoder_config("stance", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")
    mode_dir = output_root / dataset_id / artifact_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    event_metas: list[StanceEventMeta] = []
    primary_class_names: list[str] = []
    total_rows = 0
    skipped_counts: dict[str, int] = {}
    data_root = input_root.parent
    pending_rows: list[dict[str, Any]] = []
    shard_id = 0
    build_started_at = time.perf_counter()
    event_loop_elapsed = 0.0
    encode_elapsed_total = 0.0
    write_elapsed_total = 0.0
    effective_encode_batch_size = config.encode_batch_size_value(encode_batch_size_override)

    tokenizer = None
    model = None
    torch_device = None
    resolved_device = device
    device_fallback_reason = None
    use_encoder = dataset_id not in {"Twitter15", "Twitter16"}

    try:
        if use_encoder:
            import torch
            from transformers import AutoModel, AutoTokenizer

            configure_torch_runtime(tokenizer_parallelism=tokenizer_parallelism, torch_num_threads=torch_num_threads)
            resolved_device, device_fallback_reason = resolve_torch_device(device)
            torch_device = torch.device(resolved_device)
            tokenizer = AutoTokenizer.from_pretrained(config.model_name(mode))
            model = AutoModel.from_pretrained(config.model_name(mode)).to(torch_device)
            model_hidden_size = int(model.config.hidden_size)
            if config.reply_embedding_dim != model_hidden_size:
                print(
                    f"[stance_view] adapting reply_embedding_dim from "
                    f"{config.reply_embedding_dim} to {model_hidden_size}"
                )
                config.reply_embedding_dim = model_hidden_size
            model.eval()
            print(
                f"[stance_view] encoder_device={resolved_device} requested_device={device} "
                f"mode={mode} model={config.model_name(mode)} encode_batch_size={effective_encode_batch_size}"
            )
            if device_fallback_reason is not None:
                print(f"[stance_view] device_fallback_reason={device_fallback_reason}")

        for event_index, event_row in enumerate(iter_preprocessed_events(input_path)):
            event_started_at = time.perf_counter()
            if pending_rows and len(pending_rows) >= config.shard_size:
                shard_id, write_elapsed = _flush_pending_rows(
                    pending_rows,
                    mode_dir=mode_dir,
                    artifact_mode=artifact_mode,
                    shard_id=shard_id,
                    gc_interval_shards=gc_interval_shards,
                )
                write_elapsed_total += write_elapsed

            replies = (event_row.get("stance") or {}).get("reply_items") or []
            reply_by_id = {reply["post_id"]: reply for reply in replies}
            source_post_id = str((event_row.get("stance") or {}).get("source_post_id") or "")
            source_text_bundle = ((event_row.get("content") or {}).get("text")) or {}
            kept_replies: list[dict[str, Any]] = []
            kept_reasons: dict[str, str] = {}
            if use_encoder:
                for reply in replies:
                    keep, reason = classify_reply_for_stance(dataset_id, reply)
                    if keep:
                        kept_replies.append(reply)
                        kept_reasons[str(reply["post_id"])] = reason
                    else:
                        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
            kept_replies.sort(key=lambda row: (row.get("stable_order", 10**9), row.get("depth") or 0, row.get("post_id")))
            sampled = sample_replies_deterministically(kept_replies, config.max_replies_per_event_value())
            event_labels = align_event_labels(dataset_id, event_row.get("task_targets") or {}, data_root=str(data_root))
            if not primary_class_names:
                primary_class_names = list(event_labels["primary_class_names"])
            row_start_index = total_rows
            event_shard_id = shard_id
            row_start_in_shard = len(pending_rows)

            relation_embeddings = np.zeros((len(sampled.selected_replies), config.reply_embedding_dim_value(mode)), dtype=np.float32)
            semantic_feature_rows: list[dict[str, float]] = [{} for _ in sampled.selected_replies]
            if use_encoder and sampled.selected_replies:
                relation_texts = []
                reply_texts = []
                source_texts = []
                parent_texts_existing: list[str] = []
                parent_existing_positions: list[int] = []
                for idx, reply in enumerate(sampled.selected_replies):
                    parent_reply = reply_by_id.get(reply.get("parent_post_id"))
                    relation_texts.append(build_source_aware_text(source_text_bundle, reply.get("text") or {}, config, mode))
                    reply_texts.append(normalize_social_text(_text_value(reply.get("text"))))
                    source_texts.append(normalize_social_text(_text_value(source_text_bundle)))
                    parent_text = normalize_social_text(_text_value(parent_reply.get("text") if parent_reply else None))
                    if parent_text:
                        parent_existing_positions.append(idx)
                        parent_texts_existing.append(parent_text)
                encode_started_at = time.perf_counter()
                relation_embeddings = _encode_texts(relation_texts, tokenizer, model, config.max_length(mode), effective_encode_batch_size, torch_device)
                reply_text_embeddings = _encode_texts(reply_texts, tokenizer, model, config.max_length(mode), effective_encode_batch_size, torch_device)
                source_text_embeddings = _encode_texts(source_texts, tokenizer, model, config.max_length(mode), effective_encode_batch_size, torch_device)
                parent_text_embeddings = np.zeros_like(reply_text_embeddings)
                if parent_texts_existing:
                    encoded_parents = _encode_texts(parent_texts_existing, tokenizer, model, config.max_length(mode), effective_encode_batch_size, torch_device)
                    for parent_idx, row_idx in enumerate(parent_existing_positions):
                        parent_text_embeddings[row_idx] = encoded_parents[parent_idx]
                encode_elapsed_total += time.perf_counter() - encode_started_at
                reply_source_cosine = _cosine_similarity(reply_text_embeddings, source_text_embeddings)
                reply_parent_cosine = _cosine_similarity(reply_text_embeddings, parent_text_embeddings)
                parent_source_cosine = _cosine_similarity(parent_text_embeddings, source_text_embeddings)
                semantic_feature_rows = [
                    {
                        "reply_source_cosine": float(reply_source_cosine[idx]),
                        "reply_parent_cosine": float(reply_parent_cosine[idx]),
                        "parent_source_cosine": float(parent_source_cosine[idx]),
                    }
                    for idx in range(len(sampled.selected_replies))
                ]

            for reply_idx, reply in enumerate(sampled.selected_replies):
                parent_reply = reply_by_id.get(reply.get("parent_post_id"))
                parent_text_bundle = parent_reply.get("text") if parent_reply else None
                label_name, label_id, label_mask = align_reply_label(None)
                context_cat = build_context_cat(reply)
                context_dense = build_context_dense(reply, source_text_bundle, parent_text_bundle, semantic_feature_rows[reply_idx])
                pending_rows.append(
                    {
                        "event_id": event_row["event_id"],
                        "dataset_id": dataset_id,
                        "event_index": event_index,
                        "reply_embedding": relation_embeddings[reply_idx],
                        "context_cat": context_cat,
                        "context_dense": context_dense,
                        "reply_label_id": label_id,
                        "reply_label_mask": label_mask,
                        "event_sampled_reply_count": sampled.sampled_reply_count,
                        "event_truncation_flag": 1 if sampled.truncated else 0,
                    }
                )
                total_rows += 1
            row_end_in_shard = len(pending_rows)

            event_metas.append(
                StanceEventMeta(
                    event_index=event_index,
                    event_id=event_row["event_id"],
                    dataset_id=dataset_id,
                    platform=event_row["platform"],
                    canonical_split=event_row["canonical_split"],
                    raw_split_source=event_row.get("raw_split_source"),
                    source_post_id=source_post_id,
                    source_text=str(source_text_bundle.get("norm_light") or source_text_bundle.get("raw") or ""),
                    original_usable_reply_count=sampled.original_usable_reply_count,
                    sampled_reply_count=sampled.sampled_reply_count,
                    truncated=sampled.truncated,
                    row_start_index=row_start_index,
                    row_end_index=total_rows,
                    shard_id=event_shard_id,
                    row_start_in_shard=row_start_in_shard,
                    row_end_in_shard=row_end_in_shard,
                    primary_task_name=str(event_labels["primary_task_name"]),
                    primary_metric_name=str(event_labels["primary_metric_name"]),
                    primary_label_name=event_labels["primary_label_name"],
                    primary_label_id=int(event_labels["primary_label_id"]),
                    observed_view_mask=event_row.get("observed_view_mask"),
                    view_availability=event_row.get("view_availability"),
                )
            )
            event_loop_elapsed += time.perf_counter() - event_started_at
            if (event_index + 1) % LOG_EVERY_EVENTS == 0:
                print(
                    f"[stance_view] processed_events={event_index + 1} "
                    f"encoded_rows={total_rows} pending_rows={len(pending_rows)} shards={shard_id}"
                )

        if pending_rows:
            shard_id, write_elapsed = _flush_pending_rows(
                pending_rows,
                mode_dir=mode_dir,
                artifact_mode=artifact_mode,
                shard_id=shard_id,
                gc_interval_shards=gc_interval_shards,
            )
            write_elapsed_total += write_elapsed
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch_device is not None:
            try:
                import torch

                if str(torch_device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except ModuleNotFoundError:
                pass
        gc.collect()
    total_elapsed = time.perf_counter() - build_started_at

    manifest = {
        "dataset_id": dataset_id,
        "mode": artifact_mode,
        "config": config.as_dict(),
        "event_count": len(event_metas),
        "encoded_reply_count": total_rows,
        "event_metas": [meta.__dict__ for meta in event_metas],
        "skipped_counts": skipped_counts,
        "timing": {
            "event_loop_sec": event_loop_elapsed,
            "encode_sec": encode_elapsed_total,
            "write_sec": write_elapsed_total,
            "total_sec": total_elapsed,
        },
        "runtime": {
            "requested_device": device,
            "resolved_device": resolved_device,
            "device_fallback_reason": device_fallback_reason,
            "tokenizer_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "auto"),
            "torch_num_threads": torch_num_threads,
            "gc_interval_shards": gc_interval_shards,
            "encode_batch_size": effective_encode_batch_size,
        },
    }
    write_stance_manifest(mode_dir / f"stance_manifest_{artifact_mode}.json", manifest)
    write_stance_label_map(
        mode_dir / f"stance_label_map_{artifact_mode}.json",
        {
            "primary_task_name": event_metas[0].primary_task_name if event_metas else None,
            "primary_metric_name": event_metas[0].primary_metric_name if event_metas else None,
            "primary_class_names": primary_class_names,
            "primary_label_4way_reference": PRIMARY_LABEL_4WAY_NAMES,
            "primary_label_2way_reference": PRIMARY_LABEL_2WAY_NAMES,
            "reply_stance": STANCE_LABEL_NAMES,
            "missing_label_id": -1,
        },
    )




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/stance_view")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer_parallelism", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--gc_interval_shards", type=int, default=DEFAULT_GC_INTERVAL_SHARDS)
    parser.add_argument("--encode_batch_size_override", type=int, default=None)
    args = parser.parse_args()
    config = resolve_encoder_config("stance", args.dataset_id, "full")
    build_stance_arrays(
        args.dataset_id,
        "full",
        Path(args.input_root),
        Path(args.output_root),
        device=args.device,
        config=config,
        tokenizer_parallelism=args.tokenizer_parallelism,
        torch_num_threads=args.torch_num_threads,
        gc_interval_shards=args.gc_interval_shards,
        encode_batch_size_override=args.encode_batch_size_override,
    )


if __name__ == "__main__":
    main()
