from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.resolver import resolve_encoder_config
from preprocess.preprocess_events import build_default_args as build_preprocess_event_args
from preprocess.preprocess_events import preprocess_events
from models.encoders.content_view.artifacts import EVENT_INDEX_FILENAME
from models.encoders.content_view.pipelines.build_content_arrays import build_arrays as build_content_arrays
from models.encoders.content_view.pipelines.build_content_index import build_index as build_content_index
from models.encoders.evidence_view.pipelines.build_evidence_arrays import build_evidence_arrays
from models.encoders.source_view.pipelines.build_source_arrays import build_source_arrays
from models.fusion.config import FusionConfig
from utils.logging_utils import get_logger
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME


LOGGER = get_logger("fusion_artifacts")

ARTIFACT_METADATA_FILENAME = "artifact_metadata.json"
ARTIFACT_METADATA_VERSION = 1
ARTIFACT_BUILDER_VERSION = "2026-05-24-c"
EVENT_INDEX_ARTIFACT_KEY = "event_index"
CONTENT_ARRAYS_ARTIFACT_KEY = "content"
VIEW_ARTIFACT_KEYS = ("stance", "propagation", "evidence", "source")
ALL_ARTIFACT_KEYS = (EVENT_INDEX_ARTIFACT_KEY, CONTENT_ARRAYS_ARTIFACT_KEY, *VIEW_ARTIFACT_KEYS)


@dataclass(frozen=True)
class EnsureArtifactsOptions:
    stance_device: str = "auto"
    stance_tokenizer_parallelism: str = "auto"
    stance_torch_num_threads: int | None = None
    stance_gc_interval_shards: int = 8
    stance_encode_batch_size_override: int | None = None


@dataclass(frozen=True)
class ArtifactEnsureResult:
    artifact_key: str
    status: str
    reason: str
    fingerprint: str
    metadata_path: str
    output_dir: str


def _stable_payload_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metadata_version": ARTIFACT_METADATA_VERSION, "artifacts": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("metadata_version", ARTIFACT_METADATA_VERSION)
    payload.setdefault("artifacts", {})
    return payload


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_metadata_entry(mode_dir: Path, artifact_key: str, entry: dict[str, Any]) -> None:
    metadata_path = mode_dir / ARTIFACT_METADATA_FILENAME
    payload = _read_metadata(metadata_path)
    payload["artifacts"][artifact_key] = entry
    _write_metadata(metadata_path, payload)


def _metadata_entry(mode_dir: Path, artifact_key: str) -> dict[str, Any] | None:
    payload = _read_metadata(mode_dir / ARTIFACT_METADATA_FILENAME)
    entry = payload.get("artifacts", {}).get(artifact_key)
    return entry if isinstance(entry, dict) else None


def _mode_dir(data_root: Path, artifact_family: str, dataset_id: str, artifact_mode: str) -> Path:
    return data_root / artifact_family / dataset_id / artifact_mode


def _event_index_path(data_root: Path, dataset_id: str, artifact_mode: str) -> Path:
    return _mode_dir(data_root, "content_view", dataset_id, artifact_mode) / EVENT_INDEX_FILENAME


def _preprocessed_input_info(input_root: Path, dataset_id: str) -> dict[str, Any]:
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    return {
        "input_root": str(input_root),
        "preprocessed_events_path": str(input_path),
        "preprocessed_events_exists": input_path.exists(),
    }


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
    return Path("data/preprocessed")


def _load_event_ids(event_index_path: Path) -> list[str]:
    import numpy as np

    with np.load(event_index_path, allow_pickle=True) as data:
        raw = data["event_ids"].tolist()
    return [item.item() if hasattr(item, "item") else item for item in raw]


def compute_event_spine_hash(event_index_path: Path) -> str:
    return _stable_payload_hash({"event_ids": _load_event_ids(event_index_path)})


def resolve_required_artifacts(active_views: list[str], config: FusionConfig) -> list[str]:
    required = [EVENT_INDEX_ARTIFACT_KEY]
    if "content" in active_views:
        required.append(CONTENT_ARRAYS_ARTIFACT_KEY)
    if "stance" in active_views:
        required.append("stance")
    if "propagation" in active_views:
        required.append("propagation")
    if "evidence" in active_views:
        required.append("evidence")
    if "source" in active_views:
        required.append("source")
    _ = config
    return required


def _artifact_family_and_view_name(artifact_key: str, config: FusionConfig) -> tuple[str, str]:
    if artifact_key == EVENT_INDEX_ARTIFACT_KEY:
        return "content_view", "content"
    if artifact_key == CONTENT_ARRAYS_ARTIFACT_KEY:
        return "content_view", "content"
    if artifact_key == "stance":
        return "stance_view", "stance"
    if artifact_key == "propagation":
        return "propagation_view", "propagation"
    if artifact_key == "evidence":
        return "evidence_view", "evidence"
    if artifact_key == "source":
        return "source_view", "source"
    raise KeyError(f"Unsupported artifact key: {artifact_key}")


def _artifact_output_files(artifact_key: str, dataset_id: str, artifact_mode: str, config: FusionConfig) -> list[str]:
    family, _ = _artifact_family_and_view_name(artifact_key, config)
    mode_dir = _mode_dir(Path(""), family, dataset_id, artifact_mode)
    if artifact_key == EVENT_INDEX_ARTIFACT_KEY:
        return [str(mode_dir / EVENT_INDEX_FILENAME)]
    if artifact_key == CONTENT_ARRAYS_ARTIFACT_KEY:
        return [str(mode_dir / EVENT_INDEX_FILENAME), str(mode_dir / f"content_shard_{artifact_mode}_*.npz")]
    if artifact_key == "stance":
        return [str(mode_dir / f"stance_manifest_{artifact_mode}.json"), str(mode_dir / f"stance_shard_{artifact_mode}_*.npz")]
    if artifact_key == "propagation":
        return [str(mode_dir / f"propagation_manifest_{artifact_mode}.json"), str(mode_dir / f"propagation_shard_{artifact_mode}_*.npz")]
    if artifact_key == "evidence":
        return [str(mode_dir / f"evidence_manifest_{artifact_mode}.json"), str(mode_dir / f"evidence_shard_{artifact_mode}_*.npz")]
    if artifact_key == "source":
        return [str(mode_dir / f"source_manifest_{artifact_mode}.json"), str(mode_dir / f"source_shard_{artifact_mode}_*.npz")]
    raise KeyError(f"Unsupported artifact key: {artifact_key}")


def _artifact_exists(mode_dir: Path, artifact_key: str, artifact_mode: str, config: FusionConfig) -> bool:
    if artifact_key == EVENT_INDEX_ARTIFACT_KEY:
        return (mode_dir / EVENT_INDEX_FILENAME).exists()
    if artifact_key == CONTENT_ARRAYS_ARTIFACT_KEY:
        return (mode_dir / EVENT_INDEX_FILENAME).exists() and any(mode_dir.glob(f"content_shard_{artifact_mode}_*.npz"))
    if artifact_key == "stance":
        return (mode_dir / f"stance_manifest_{artifact_mode}.json").exists() and any(mode_dir.glob(f"stance_shard_{artifact_mode}_*.npz"))
    if artifact_key == "propagation":
        return (mode_dir / f"propagation_manifest_{artifact_mode}.json").exists() and any(mode_dir.glob(f"propagation_shard_{artifact_mode}_*.npz"))
    if artifact_key == "evidence":
        return (mode_dir / f"evidence_manifest_{artifact_mode}.json").exists() and any(mode_dir.glob(f"evidence_shard_{artifact_mode}_*.npz"))
    if artifact_key == "source":
        return (mode_dir / f"source_manifest_{artifact_mode}.json").exists() and any(mode_dir.glob(f"source_shard_{artifact_mode}_*.npz"))
    raise KeyError(f"Unsupported artifact key: {artifact_key}")


def _fingerprint_inputs(
    artifact_key: str,
    *,
    dataset_id: str,
    mode: str,
    artifact_mode: str,
    input_root: Path,
    config: FusionConfig,
    options: EnsureArtifactsOptions,
    event_spine_hash: str | None,
) -> dict[str, Any]:
    artifact_family, view_name = _artifact_family_and_view_name(artifact_key, config)
    view_config = resolve_encoder_config(view_name, dataset_id, mode)
    payload: dict[str, Any] = {
        "artifact_key": artifact_key,
        "artifact_family": artifact_family,
        "view_name": view_name,
        "dataset_id": dataset_id,
        "mode": mode,
        "artifact_mode": artifact_mode,
        "view_config": view_config.as_dict(),
        "builder_version": ARTIFACT_BUILDER_VERSION,
        "input": _preprocessed_input_info(input_root, dataset_id),
    }
    if artifact_key in {CONTENT_ARRAYS_ARTIFACT_KEY, "stance", "propagation", "evidence", "source"}:
        payload["event_spine_hash"] = event_spine_hash
    if artifact_key == "stance":
        payload["stance_encode_batch_size_override"] = options.stance_encode_batch_size_override
    return payload


def _build_artifact(
    artifact_key: str,
    *,
    dataset_id: str,
    mode: str,
    input_root: Path,
    data_root: Path,
    config: FusionConfig,
    options: EnsureArtifactsOptions,
) -> None:
    if artifact_key == EVENT_INDEX_ARTIFACT_KEY:
        content_config = resolve_encoder_config("content", dataset_id, mode)
        build_content_index(dataset_id, mode, input_root, data_root / "content_view", config=content_config)
        return
    if artifact_key == CONTENT_ARRAYS_ARTIFACT_KEY:
        content_config = resolve_encoder_config("content", dataset_id, mode)
        build_content_arrays(dataset_id, mode, input_root, data_root / "content_view", config=content_config)
        return
    if artifact_key == "stance":
        stance_config = resolve_encoder_config("stance", dataset_id, mode)
        from models.encoders.stance_view.pipelines.build_stance_arrays import build_stance_arrays
        build_stance_arrays(
            dataset_id,
            mode,
            input_root,
            data_root / "stance_view",
            device=options.stance_device,
            config=stance_config,
            tokenizer_parallelism=options.stance_tokenizer_parallelism,
            torch_num_threads=options.stance_torch_num_threads,
            gc_interval_shards=options.stance_gc_interval_shards,
            encode_batch_size_override=options.stance_encode_batch_size_override,
        )
        return
    if artifact_key == "propagation":
        propagation_config = resolve_encoder_config("propagation", dataset_id, mode)
        from models.encoders.propagation_view.pipelines.build_propagation_arrays import build_propagation_arrays
        build_propagation_arrays(
            dataset_id,
            mode,
            input_root,
            data_root / "propagation_view",
            config=propagation_config,
        )
        return
    if artifact_key == "evidence":
        evidence_config = resolve_encoder_config("evidence", dataset_id, mode)
        build_evidence_arrays(dataset_id, mode, input_root, data_root / "evidence_view", config=evidence_config)
        return
    if artifact_key == "source":
        source_config = resolve_encoder_config("source", dataset_id, mode)
        build_source_arrays(dataset_id, mode, input_root, data_root / "source_view", config=source_config)
        return
    raise KeyError(f"Unsupported artifact key: {artifact_key}")


def _ensure_single_artifact(
    artifact_key: str,
    *,
    dataset_id: str,
    mode: str,
    input_root: Path,
    data_root: Path,
    config: FusionConfig,
    options: EnsureArtifactsOptions,
    event_spine_hash: str | None,
) -> ArtifactEnsureResult:
    artifact_mode = config.artifact_mode()
    artifact_family, _ = _artifact_family_and_view_name(artifact_key, config)
    mode_dir = _mode_dir(data_root, artifact_family, dataset_id, artifact_mode)
    metadata_path = mode_dir / ARTIFACT_METADATA_FILENAME
    fingerprint_input = _fingerprint_inputs(
        artifact_key,
        dataset_id=dataset_id,
        mode=mode,
        artifact_mode=artifact_mode,
        input_root=input_root,
        config=config,
        options=options,
        event_spine_hash=event_spine_hash,
    )
    fingerprint = _stable_payload_hash(fingerprint_input)
    metadata_entry = _metadata_entry(mode_dir, artifact_key)
    exists = _artifact_exists(mode_dir, artifact_key, artifact_mode, config)
    reason = "cache_hit"
    status = "reused"
    if not exists:
        status = "rebuilt"
        reason = "missing"
    elif metadata_entry is None:
        status = "rebuilt"
        reason = "missing_metadata"
    elif metadata_entry.get("fingerprint") != fingerprint:
        status = "rebuilt"
        reason = "fingerprint_mismatch"
    elif artifact_key != EVENT_INDEX_ARTIFACT_KEY and metadata_entry.get("event_spine_hash") != event_spine_hash:
        status = "rebuilt"
        reason = "event_spine_mismatch"

    if status == "rebuilt":
        _build_artifact(
            artifact_key,
            dataset_id=dataset_id,
            mode=mode,
            input_root=input_root,
            data_root=data_root,
            config=config,
            options=options,
        )
        artifact_event_spine_hash = compute_event_spine_hash(_event_index_path(data_root, dataset_id, artifact_mode))
        _update_metadata_entry(
            mode_dir,
            artifact_key,
            {
                "artifact_key": artifact_key,
                "artifact_family": artifact_family,
                "fingerprint": fingerprint,
                "fingerprint_input": fingerprint_input,
                "event_spine_hash": artifact_event_spine_hash if artifact_key != EVENT_INDEX_ARTIFACT_KEY else artifact_event_spine_hash,
                "builder_version": ARTIFACT_BUILDER_VERSION,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "output_files": _artifact_output_files(artifact_key, dataset_id, artifact_mode, config),
            },
        )

    return ArtifactEnsureResult(
        artifact_key=artifact_key,
        status=status,
        reason=reason,
        fingerprint=fingerprint,
        metadata_path=str(metadata_path),
        output_dir=str(mode_dir),
    )


def ensure_required_artifacts(
    *,
    dataset_id: str,
    mode: str,
    active_views: list[str],
    config: FusionConfig,
    data_root: str | Path = "data",
    input_root: str | Path = "data/preprocessed",
    options: EnsureArtifactsOptions | None = None,
) -> list[ArtifactEnsureResult]:
    data_root = Path(data_root)
    input_root = _ensure_preprocessed_events(dataset_id, Path(input_root))
    options = options or EnsureArtifactsOptions()
    results: list[ArtifactEnsureResult] = []
    required_artifacts = resolve_required_artifacts(active_views, config)
    current_event_spine_hash: str | None = None
    for artifact_key in required_artifacts:
        result = _ensure_single_artifact(
            artifact_key,
            dataset_id=dataset_id,
            mode=mode,
            input_root=input_root,
            data_root=data_root,
            config=config,
            options=options,
            event_spine_hash=current_event_spine_hash,
        )
        results.append(result)
        if artifact_key == EVENT_INDEX_ARTIFACT_KEY:
            current_event_spine_hash = compute_event_spine_hash(_event_index_path(data_root, dataset_id, config.artifact_mode()))
        LOGGER.info(
            "artifact ensure: dataset=%s mode=%s artifact=%s status=%s reason=%s",
            dataset_id,
            mode,
            artifact_key,
            result.status,
            result.reason,
        )
    return results
