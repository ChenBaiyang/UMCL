from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from config.resolver import resolve_encoder_config
from models.encoders.content_view.artifacts import EVENT_INDEX_FILENAME
from models.fusion.accessors import NpzShardCache, RangeShardAccessor, RowShardAccessor
from models.fusion.label_space import primary_task_spec_from_event_index
from utils.jsonl import read_jsonl


VIEW_ORDER = ("content", "stance", "propagation", "evidence", "source")


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return value


class FiveViewDataset(Dataset):
    def __init__(
        self,
        dataset_id: str,
        mode: str = "full",
        data_root: str | Path = "data",
        cache_max_open_shards: int = 16,
        required_views: set[str] | None = None,
        load_bucket_metadata: bool = False,
    ) -> None:
        self.dataset_id = dataset_id
        self.mode = mode
        self.data_root = Path(data_root)
        self.cache = NpzShardCache(max_open_shards=cache_max_open_shards)

        self.content_cfg = resolve_encoder_config("content", dataset_id, mode)
        self.stance_cfg = resolve_encoder_config("stance", dataset_id, mode)
        self.propagation_cfg = resolve_encoder_config("propagation", dataset_id, mode)
        self.evidence_cfg = resolve_encoder_config("evidence", dataset_id, mode)
        self.source_cfg = resolve_encoder_config("source", dataset_id, mode)
        self.artifact_mode = self.content_cfg.artifact_mode(mode)
        self.use_stance_cached_embeddings = False
        self.required_views = set(VIEW_ORDER) if required_views is None else set(required_views)
        self.load_bucket_metadata = bool(load_bucket_metadata)
        self._evidence_ref_slices_cache: dict[int, list[np.ndarray]] = {}

        self._reset_optional_state()
        self._load_event_index()
        if "stance" in self.required_views:
            self._load_stance_manifest()
            self._load_stance_embedding_cache()
        if "propagation" in self.required_views:
            self._load_propagation_manifest(load_accessor=True)
        elif self.load_bucket_metadata:
            self._load_propagation_manifest(load_accessor=False)
        if "evidence" in self.required_views:
            self._load_evidence_meta()
        if "source" in self.required_views:
            self._load_source_meta()
        self._validate_event_alignment()

    def _reset_optional_state(self) -> None:
        self.stance_manifest = None
        self.stance_event_metas: list[dict[str, Any]] = []
        self.stance_ranges: list[tuple[int, int]] = []
        self.stance_accessor = None
        self.stance_reply_embedding_dim = int(self.stance_cfg.reply_embedding_dim_value(self.mode))
        self.stance_embedding_manifest = None
        self.stance_embedding_accessor = None

        self.propagation_manifest = None
        self.propagation_event_metas: list[dict[str, Any]] = []
        self.propagation_accessor = None
        self.propagation_artifact_family = "propagation_view"
        self.propagation_continuous_feature_names: list[str] = []
        self.propagation_categorical_feature_names: list[str] = []
        self.propagation_event_summary_feature_names: list[str] = []
        self.propagation_temporal_profile_feature_names: list[str] = []

        self.evidence_event_meta: list[dict[str, Any]] = []
        self.evidence_manifest = None
        self.evidence_mode_dir = None
        self.evidence_accessor = None
        self.evidence_shard_size = 0
        self.evidence_shard_prefix = "evidence_shard"

        self.source_event_meta: list[dict[str, Any]] = []
        self.source_manifest = None
        self.source_accessor = None
        self.source_shard_size = 0

    def _load_event_index(self) -> None:
        path = self.data_root / "content_view" / self.dataset_id / self.artifact_mode / EVENT_INDEX_FILENAME
        with np.load(path, allow_pickle=True) as data:
            self.event_index = {key: data[key] for key in data.files}
        self.content_accessor = RowShardAccessor(
            path.parent,
            f"content_shard_{self.artifact_mode}_" + "{shard_id:04d}.npz",
            self.cache,
            keys=("input_ids", "attention_mask", "surface_features", "quality_flags", "primary_label_ids"),
        )
        self.event_ids = [_to_scalar(v) for v in self.event_index["event_ids"]]
        self.primary_task_spec = primary_task_spec_from_event_index(self.dataset_id, self.event_index)

    def _load_stance_manifest(self) -> None:
        path = self._stance_mode_dir() / f"stance_manifest_{self.artifact_mode}.json"
        self.stance_manifest = json.loads(path.read_text(encoding="utf-8"))
        self.stance_event_metas = self.stance_manifest["event_metas"]
        cumulative = 0
        self.stance_ranges = []
        for meta in self.stance_event_metas:
            count = self._stance_reply_count(meta)
            self.stance_ranges.append((cumulative, cumulative + count))
            cumulative += count
        self.stance_accessor = RangeShardAccessor(
            path.parent,
            f"stance_shard_{self.artifact_mode}_" + "{shard_id:04d}.npz",
            int(self.stance_manifest["config"]["shard_size"]),
            self.cache,
            keys=(
                "reply_embedding",
                "context_cat",
                "context_dense",
                "reply_label_ids",
                "reply_label_mask",
                "event_sampled_reply_count",
                "event_truncation_flag",
            ),
        )
        self.stance_reply_embedding_dim = self._resolve_stance_reply_embedding_dim(path.parent)

    def _resolve_stance_reply_embedding_dim(self, mode_dir: Path) -> int:
        shard_paths = sorted(mode_dir.glob(f"stance_shard_{self.artifact_mode}_*.npz"))
        shard_path = shard_paths[0] if shard_paths else None
        if shard_path is not None:
            with np.load(shard_path, allow_pickle=False) as payload:
                reply_embedding = payload.get("reply_embedding")
                if reply_embedding is not None and reply_embedding.ndim == 2:
                    return int(reply_embedding.shape[-1])
        return int(self.stance_cfg.reply_embedding_dim_value(self.mode))

    def _stance_mode_dir(self) -> Path:
        return self.data_root / "stance_view" / self.dataset_id / self.artifact_mode

    def _stance_reply_count(self, meta: dict[str, Any]) -> int:
        return int(meta["sampled_reply_count"])

    def _load_propagation_manifest(self, *, load_accessor: bool) -> None:
        self.propagation_artifact_family = "propagation_view"
        mode_dir = self.data_root / self.propagation_artifact_family / self.dataset_id / self.artifact_mode
        path = mode_dir / f"propagation_manifest_{self.artifact_mode}.json"
        if not path.exists():
            legacy_mode_dir = self.data_root / self.propagation_artifact_family / self.dataset_id
            legacy_path = legacy_mode_dir / f"propagation_manifest_{self.artifact_mode}.json"
            if legacy_path.exists():
                mode_dir = legacy_mode_dir
                path = legacy_path
        self.propagation_manifest = json.loads(path.read_text(encoding="utf-8"))
        self.propagation_event_metas = self.propagation_manifest["event_meta"]
        self.propagation_continuous_feature_names = list(self.propagation_manifest.get("active_continuous_feature_names") or [])
        self.propagation_categorical_feature_names = list(self.propagation_manifest.get("active_categorical_feature_names") or [])
        self.propagation_event_summary_feature_names = list(self.propagation_manifest.get("event_summary_feature_names") or [])
        self.propagation_temporal_profile_feature_names = list(self.propagation_manifest.get("temporal_profile_feature_names") or [])
        if not self.propagation_continuous_feature_names:
            self.propagation_continuous_feature_names = [f"continuous_{idx}" for idx in range(6)]
        if not self.propagation_categorical_feature_names:
            self.propagation_categorical_feature_names = [f"categorical_{idx}" for idx in range(11)]
        if self.load_bucket_metadata and self.propagation_event_metas:
            if "node_count_bucket_id" not in self.propagation_event_metas[0]:
                raise ValueError(
                    f"Propagation manifest for dataset={self.dataset_id}, mode={self.mode} is missing "
                    "bucket metadata. Please rebuild propagation artifacts."
                )
        if load_accessor:
            self.propagation_accessor = RangeShardAccessor(
                mode_dir,
                f"propagation_shard_{self.artifact_mode}_" + "{shard_id:04d}.npz",
                int(self.propagation_manifest["config_snapshot"]["shard_size"]),
                self.cache,
                keys=(
                    "node_features_continuous",
                    "node_features_categorical",
                    "node_masks",
                    "parent_index",
                    "node_sampling_role_ids",
                    "event_summary_features",
                    "temporal_profile_features",
                ),
            )

    def _load_evidence_meta(self) -> None:
        mode_dir = self.data_root / "evidence_view" / self.dataset_id / self.artifact_mode
        event_meta_path = mode_dir / "evidence_event_meta.jsonl"
        manifest_path = mode_dir / f"evidence_manifest_{self.artifact_mode}.json"
        self.evidence_event_meta = read_jsonl(event_meta_path)
        self.evidence_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.evidence_mode_dir = mode_dir
        self.evidence_shard_size = int(self.evidence_manifest["config_snapshot"]["shard_size"])
        self.evidence_shard_prefix = "evidence_shard"
        self.evidence_accessor = RowShardAccessor(
            mode_dir,
            f"{self.evidence_shard_prefix}_{self.artifact_mode}_" + "{shard_id:04d}.npz",
            self.cache,
            keys=("summary_features", "summary_quality_flags", "primary_label_ids", "event_ref_row_features", "event_ref_row_mask"),
        )
        self._evidence_ref_slices_cache.clear()

    def _load_source_meta(self) -> None:
        mode_dir = self.data_root / "source_view" / self.dataset_id / self.artifact_mode
        self.source_event_meta = read_jsonl(mode_dir / "source_event_meta.jsonl")
        self.source_manifest = json.loads((mode_dir / f"source_manifest_{self.artifact_mode}.json").read_text(encoding="utf-8"))
        self.source_accessor = RowShardAccessor(
            mode_dir,
            f"source_shard_{self.artifact_mode}_" + "{shard_id:04d}.npz",
            self.cache,
            keys=("summary_features", "summary_quality_flags", "categorical_features", "primary_label_ids"),
        )
        self.source_shard_size = int(self.source_manifest["config_snapshot"]["shard_size"])

    def _load_stance_embedding_cache(self) -> None:
        self.stance_embedding_manifest = None
        self.stance_embedding_accessor = None
        return

    def _evidence_ref_slices_for_shard(self, shard_id: int) -> list[np.ndarray]:
        cached = self._evidence_ref_slices_cache.get(shard_id)
        if cached is not None:
            return cached
        if self.evidence_mode_dir is None:
            raise RuntimeError("Evidence mode dir is not initialized.")
        shard_path = self.evidence_mode_dir / f"{self.evidence_shard_prefix}_{self.artifact_mode}_{shard_id:04d}.npz"
        shard_payload = self.cache.load_fields(shard_path, ("summary_features", "ref_row_event_index"))
        n_rows = int(np.asarray(shard_payload["summary_features"]).shape[0])
        ref_index = np.asarray(shard_payload["ref_row_event_index"], dtype=np.int32)
        if ref_index.size == 0:
            slices = [np.zeros((0,), dtype=np.int32) for _ in range(n_rows)]
        else:
            buckets: list[list[int]] = [[] for _ in range(n_rows)]
            for row_idx, event_idx in enumerate(ref_index.tolist()):
                if 0 <= event_idx < n_rows:
                    buckets[event_idx].append(row_idx)
            slices = [np.asarray(rows, dtype=np.int32) for rows in buckets]
        self._evidence_ref_slices_cache[shard_id] = slices
        return slices

    def _validate_event_alignment(self) -> None:
        checks: list[list[Any]] = []
        if self.stance_event_metas:
            checks.append([_to_scalar(meta["event_id"]) for meta in self.stance_event_metas])
        if self.propagation_event_metas:
            checks.append([_to_scalar(meta["event_id"]) for meta in self.propagation_event_metas])
        if self.evidence_event_meta:
            checks.append([_to_scalar(meta["event_id"]) for meta in self.evidence_event_meta])
        if self.source_event_meta:
            checks.append([_to_scalar(meta["event_id"]) for meta in self.source_event_meta])
        for other in checks:
            if other != self.event_ids:
                raise ValueError(f"Cross-view event order mismatch detected for dataset {self.dataset_id}.")

    @staticmethod
    def _event_local_slice(meta: dict[str, Any], start_key: str, end_key: str) -> tuple[int, int, int] | None:
        if "shard_id" not in meta:
            return None
        if start_key not in meta or end_key not in meta:
            return None
        return int(meta["shard_id"]), int(meta[start_key]), int(meta[end_key])

    def __len__(self) -> int:
        return len(self.event_ids)

    @property
    def feature_dims(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        if "content" in self.required_views:
            dims["content_surface"] = len(self.content_cfg.surface_feature_names)
        if "propagation" in self.required_views:
            dims["propagation_continuous"] = len(self.propagation_continuous_feature_names)
            dims["propagation_categorical"] = len(self.propagation_categorical_feature_names)
            dims["propagation_event_summary"] = len(self.propagation_event_summary_feature_names)
            dims["propagation_temporal_profile"] = len(self.propagation_temporal_profile_feature_names)
        if "evidence" in self.required_views and self.evidence_manifest is not None:
            dims["evidence_summary"] = len(self.evidence_manifest["summary_feature_names"])
            dims["evidence_quality"] = len(self.evidence_manifest["summary_quality_flag_names"])
            dims["evidence_ref"] = len(self.evidence_manifest["ref_row_feature_names"])
        if "source" in self.required_views:
            dims["source_summary"] = len(self.source_cfg.summary_feature_names)
            dims["source_quality"] = len(self.source_cfg.summary_quality_flag_names)
            dims["source_categorical"] = len(self.source_cfg.categorical_feature_names)
        if "stance" in self.required_views:
            dims["stance_reply_embedding"] = int(self.stance_reply_embedding_dim)
            dims["stance_context_cat"] = len(self.stance_cfg.context_cat_fields)
            dims["stance_context_dense"] = len(self.stance_cfg.context_dense_fields)
        return dims

    def _empty_content_item(self) -> dict[str, Any]:
        seq_len = self.content_cfg.max_length(self.mode)
        return {
            "input_ids": np.zeros((seq_len,), dtype=np.int32),
            "attention_mask": np.zeros((seq_len,), dtype=np.uint8),
            "surface_features": np.zeros((len(self.content_cfg.surface_feature_names),), dtype=np.float32),
            "quality_flags": np.zeros((5,), dtype=np.uint8),
            "primary_label_id": int(_to_scalar(self.event_index["primary_label_ids"][0])) if len(self.event_ids) else -1,
        }

    def _load_content_item(self, index: int) -> tuple[dict[str, Any], float]:
        if "content" not in self.required_views:
            item = self._empty_content_item()
            item["primary_label_id"] = int(_to_scalar(self.event_index["primary_label_ids"][index]))
            return item, 0.0
        shard_id = int(self.event_index["shard_ids"][index])
        row_in_shard = int(self.event_index["row_in_shard"][index])
        row = self.content_accessor.get_row(shard_id, row_in_shard)
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "surface_features": row["surface_features"],
            "quality_flags": row["quality_flags"],
            "primary_label_id": int(row["primary_label_ids"]),
        }, 1.0

    def _empty_stance_item(self) -> tuple[dict[str, Any], float]:
        return {
            "reply_embedding": np.zeros((0, self.stance_reply_embedding_dim), dtype=np.float32),
            "context_cat": np.zeros((0, len(self.stance_cfg.context_cat_fields)), dtype=np.int16),
            "context_dense": np.zeros((0, len(self.stance_cfg.context_dense_fields)), dtype=np.float32),
            "reply_label_ids": np.zeros((0,), dtype=np.int16),
            "reply_label_mask": np.zeros((0,), dtype=np.uint8),
            "event_sampled_reply_count": np.zeros((0,), dtype=np.int16),
            "event_truncation_flag": np.zeros((0,), dtype=np.uint8),
        }, 0.0

    def _load_stance_item(self, index: int) -> tuple[dict[str, Any], float]:
        if "stance" not in self.required_views or not self.stance_ranges or self.stance_accessor is None:
            return self._empty_stance_item()
        meta = self.stance_event_metas[index]
        event_slice = self._event_local_slice(meta, "row_start_in_shard", "row_end_in_shard")
        if event_slice is not None:
            shard_id, local_start, local_end = event_slice
            if local_end <= local_start:
                return self._empty_stance_item()
            rows = self.stance_accessor.get_slice(shard_id, local_start, local_end)
        else:
            start, end = self.stance_ranges[index]
            if end <= start:
                return self._empty_stance_item()
            rows = self.stance_accessor.get_range(start, end)
        return {
            "reply_embedding": rows["reply_embedding"],
            "context_cat": rows["context_cat"],
            "context_dense": rows["context_dense"],
            "reply_label_ids": rows["reply_label_ids"],
            "reply_label_mask": rows["reply_label_mask"],
            "event_sampled_reply_count": rows["event_sampled_reply_count"],
            "event_truncation_flag": rows["event_truncation_flag"],
        }, 1.0

    def _empty_propagation_item(self) -> tuple[dict[str, Any], float]:
        continuous_dim = len(self.propagation_continuous_feature_names)
        categorical_dim = len(self.propagation_categorical_feature_names)
        event_summary_dim = len(self.propagation_event_summary_feature_names)
        temporal_profile_dim = len(self.propagation_temporal_profile_feature_names)
        return {
            "node_features_continuous": np.zeros((0, continuous_dim), dtype=np.float32),
            "node_features_categorical": np.zeros((0, categorical_dim), dtype=np.int16),
            "node_masks": np.zeros((0,), dtype=np.uint8),
            "parent_index": np.zeros((0,), dtype=np.int32),
            "node_sampling_role_ids": np.zeros((0,), dtype=np.int16),
            "raw_n_nodes": 0,
            "n_encoded_nodes": 0,
            "max_nodes_per_event": self.propagation_cfg.max_nodes_per_event_value(),
            "event_summary_features": np.zeros((event_summary_dim,), dtype=np.float32),
            "temporal_profile_features": np.zeros((temporal_profile_dim,), dtype=np.float32),
        }, 0.0

    def _load_propagation_item(self, index: int) -> tuple[dict[str, Any], float]:
        if "propagation" not in self.required_views or not self.propagation_event_metas or self.propagation_accessor is None:
            return self._empty_propagation_item()
        meta = self.propagation_event_metas[index]
        event_slice = self._event_local_slice(meta, "node_start_in_shard", "node_end_in_shard")
        if event_slice is not None:
            shard_id, local_start, local_end = event_slice
            if local_end <= local_start:
                return self._empty_propagation_item()
            rows = self.propagation_accessor.get_slice(shard_id, local_start, local_end)
        else:
            start = int(meta["node_start_index"])
            end = int(meta["node_end_index"])
            if end <= start:
                return self._empty_propagation_item()
            rows = self.propagation_accessor.get_range(start, end)
        return {
            "node_features_continuous": rows["node_features_continuous"],
            "node_features_categorical": rows["node_features_categorical"],
            "node_masks": rows["node_masks"],
            "parent_index": rows["parent_index"],
            "node_sampling_role_ids": rows.get(
                "node_sampling_role_ids",
                np.zeros((rows["node_features_continuous"].shape[0],), dtype=np.int16),
            ),
            "raw_n_nodes": int(meta.get("raw_n_nodes", meta.get("n_candidate_nodes", rows["node_features_continuous"].shape[0]))),
            "n_encoded_nodes": int(meta.get("n_encoded_nodes", rows["node_features_continuous"].shape[0])),
            "max_nodes_per_event": self.propagation_cfg.max_nodes_per_event_value(),
            "event_summary_features": np.asarray(
                rows.get("event_summary_features", np.zeros((len(self.propagation_event_summary_feature_names),), dtype=np.float32))[0]
                if getattr(rows.get("event_summary_features"), "ndim", 0) == 2
                else rows.get("event_summary_features", np.zeros((len(self.propagation_event_summary_feature_names),), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1),
            "temporal_profile_features": np.asarray(
                rows.get("temporal_profile_features", np.zeros((len(self.propagation_temporal_profile_feature_names),), dtype=np.float32))[0]
                if getattr(rows.get("temporal_profile_features"), "ndim", 0) == 2
                else rows.get("temporal_profile_features", np.zeros((len(self.propagation_temporal_profile_feature_names),), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1),
        }, 1.0

    def _empty_evidence_item(self) -> tuple[dict[str, Any], float]:
        summary_dim = len(self.evidence_cfg.summary_feature_names)
        quality_dim = len(self.evidence_cfg.summary_quality_flag_names)
        return {
            "summary_features": np.zeros((summary_dim,), dtype=np.float32),
            "summary_quality_flags": np.zeros((quality_dim,), dtype=np.uint8),
            "ref_row_features": np.zeros((0, 8), dtype=np.float32),
            "ref_row_mask": np.zeros((0,), dtype=np.uint8),
            "primary_label_id": -1,
        }, 0.0

    def _load_evidence_item(self, index: int) -> tuple[dict[str, Any], float]:
        if "evidence" not in self.required_views or self.evidence_mode_dir is None or self.evidence_manifest is None or self.evidence_accessor is None:
            return self._empty_evidence_item()
        shard_id = index // self.evidence_shard_size
        row_in_shard = index % self.evidence_shard_size
        row = self.evidence_accessor.get_row(shard_id, row_in_shard)
        ref_features = row.get("event_ref_row_features")
        ref_mask = row.get("event_ref_row_mask")
        if ref_features is None or ref_mask is None:
            if self.evidence_mode_dir is None:
                raise RuntimeError("Evidence mode dir is not initialized.")
            shard_path = self.evidence_mode_dir / f"{self.evidence_shard_prefix}_{self.artifact_mode}_{shard_id:04d}.npz"
            shard_payload = self.cache.load_fields(shard_path, ("ref_row_features",))
            ref_rows = self._evidence_ref_slices_for_shard(shard_id)[row_in_shard]
            ref_dim = len(self.evidence_manifest.get("ref_row_feature_names") or [])
            if ref_rows.size:
                ref_features = np.asarray(shard_payload["ref_row_features"][ref_rows], dtype=np.float32)
                ref_mask = np.ones((ref_rows.size,), dtype=np.uint8)
            else:
                ref_features = np.zeros((0, ref_dim), dtype=np.float32)
                ref_mask = np.zeros((0,), dtype=np.uint8)
        return {
            "summary_features": row["summary_features"],
            "summary_quality_flags": row["summary_quality_flags"],
            "ref_row_features": ref_features,
            "ref_row_mask": ref_mask,
            "primary_label_id": int(row["primary_label_ids"]),
        }, 1.0

    def _empty_source_item(self) -> tuple[dict[str, Any], float]:
        return {
            "summary_features": np.zeros((len(self.source_cfg.summary_feature_names),), dtype=np.float32),
            "summary_quality_flags": np.zeros((len(self.source_cfg.summary_quality_flag_names),), dtype=np.uint8),
            "categorical_features": np.zeros((len(self.source_cfg.categorical_feature_names),), dtype=np.int16),
            "primary_label_id": -1,
            "source_mode": "inactive",
        }, 0.0

    def _load_source_item(self, index: int) -> tuple[dict[str, Any], float]:
        if "source" not in self.required_views or self.source_accessor is None:
            return self._empty_source_item()
        shard_id = index // self.source_shard_size
        row_in_shard = index % self.source_shard_size
        row = self.source_accessor.get_row(shard_id, row_in_shard)
        mode = self.source_event_meta[index].get("source_mode") or "missing"
        return {
            "summary_features": row["summary_features"],
            "summary_quality_flags": row["summary_quality_flags"],
            "categorical_features": row["categorical_features"],
            "primary_label_id": int(row["primary_label_ids"]),
            "source_mode": mode,
        }, (0.0 if mode == "missing" else 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        content_item, content_mask = self._load_content_item(index)
        stance_item, stance_mask = self._load_stance_item(index)
        propagation_item, propagation_mask = self._load_propagation_item(index)
        evidence_item, evidence_mask = self._load_evidence_item(index)
        source_item, source_mask = self._load_source_item(index)
        event_meta = self.event_index

        stance_mode = self.stance_event_metas[index].get("stance_mode", "cached_embedding") if self.stance_event_metas else "inactive"
        propagation_mode = self.propagation_event_metas[index].get("propagation_mode", "inactive") if self.propagation_event_metas else "inactive"
        evidence_mode = self.evidence_event_meta[index].get("evidence_mode", "inactive") if self.evidence_event_meta else "inactive"
        source_mode = self.source_event_meta[index].get("source_mode", "inactive") if self.source_event_meta else "inactive"

        return {
            "event_id": self.event_ids[index],
            "dataset_id": _to_scalar(event_meta["dataset_ids"][index]),
            "platform": _to_scalar(event_meta["platforms"][index]),
            "canonical_split": _to_scalar(event_meta["canonical_splits"][index]),
            "raw_split_source": None if event_meta["raw_split_sources"][index] is None else _to_scalar(event_meta["raw_split_sources"][index]),
            "content": content_item,
            "stance": stance_item,
            "propagation": propagation_item,
            "evidence": evidence_item,
            "source": source_item,
            "labels": {
                "primary_task": int(_to_scalar(event_meta["primary_label_ids"][index])),
            },
            "modality_mask": {
                "content": float(content_mask),
                "stance": float(stance_mask),
                "propagation": float(propagation_mask),
                "evidence": float(evidence_mask),
                "source": float(source_mask),
            },
            "metadata": {
                "stance_mode": stance_mode,
                "propagation_mode": propagation_mode,
                "evidence_mode": evidence_mode,
                "source_mode": source_mode,
            },
        }

    def reply_count_at(self, index: int) -> int:
        if not self.stance_event_metas:
            return 0
        return self._stance_reply_count(self.stance_event_metas[index])

    def node_count_at(self, index: int) -> int:
        if not self.propagation_event_metas:
            return 0
        meta = self.propagation_event_metas[index]
        return max(0, int(meta["node_end_index"]) - int(meta["node_start_index"]))

    def bucket_id_at(self, index: int) -> int:
        if not self.propagation_event_metas:
            return 0
        meta = self.propagation_event_metas[index]
        if "node_count_bucket_id" not in meta:
            raise ValueError(
                f"Propagation event meta for dataset={self.dataset_id}, mode={self.mode} is missing "
                "node_count_bucket_id. Please rebuild propagation artifacts."
            )
        return int(meta["node_count_bucket_id"])

    def primary_label_at(self, index: int) -> int:
        return int(_to_scalar(self.event_index["primary_label_ids"][index]))

    def cache_stats(self) -> dict[str, int]:
        return self.cache.stats()
