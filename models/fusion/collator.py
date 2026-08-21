from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np
import torch

from models.fusion.schemas import FusionBatch


def _numpy_to_torch(array: np.ndarray, *, dtype: np.dtype | None = None) -> torch.Tensor:
    np_array = np.asarray(array, dtype=dtype)
    if not np_array.flags.writeable:
        np_array = np.array(np_array, copy=True)
    return torch.from_numpy(np_array)


NODE_SAMPLING_ROLE_ALL_KEPT = 0
NODE_SAMPLING_ROLE_SOURCE = 1
NODE_SAMPLING_ROLE_EARLY = 2
NODE_SAMPLING_ROLE_MIDDLE_POOL = 3
NODE_SAMPLING_ROLE_LATE = 4


def _slice_propagation_arrays(row: dict[str, Any], keep_indices: np.ndarray) -> dict[str, Any]:
    old_parent = np.asarray(row["parent_index"], dtype=np.int64)
    old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(keep_indices.tolist())}
    remapped_parent = np.full((keep_indices.shape[0],), -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(keep_indices.tolist()):
        parent_idx = int(old_parent[int(old_idx)])
        if parent_idx in old_to_new:
            remapped_parent[new_idx] = old_to_new[parent_idx]
    return {
        "node_features_continuous": row["node_features_continuous"][keep_indices],
        "node_features_categorical": row["node_features_categorical"][keep_indices],
        "node_masks": row["node_masks"][keep_indices],
        "parent_index": remapped_parent,
        "node_sampling_role_ids": np.asarray(row.get("node_sampling_role_ids", np.zeros((row["node_features_continuous"].shape[0],), dtype=np.int16)))[keep_indices],
        "event_summary_features": row["event_summary_features"],
        "temporal_profile_features": row["temporal_profile_features"],
    }


def _budget_counts(max_nodes: int, early_fraction: float, middle_fraction: float) -> tuple[int, int, int]:
    remaining_budget = max(0, int(max_nodes) - 1)
    early_budget = int(remaining_budget * float(early_fraction))
    middle_budget = int(remaining_budget * float(middle_fraction))
    late_budget = max(0, remaining_budget - early_budget - middle_budget)
    return early_budget, middle_budget, late_budget


def _middle_window_indices_seeded(
    middle_indices: np.ndarray,
    middle_budget: int,
    *,
    stochastic: bool,
    sampling_seed: int | None,
    sampling_epoch: int,
) -> np.ndarray:
    if middle_budget <= 0 or middle_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if middle_indices.size <= middle_budget:
        return middle_indices.astype(np.int64, copy=False)
    max_start = int(middle_indices.size - middle_budget)
    if stochastic:
        feasible_start_count = max_start + 1
        if sampling_seed is None:
            raise ValueError("stochastic propagation node sampling requires propagation_sampling_seed.")
        _ = sampling_seed
        stride = max(1, int(middle_budget * 0.05))
        start = (max(0, int(sampling_epoch) - 1) * stride) % feasible_start_count
    else:
        start = max_start // 2
    return middle_indices[start : start + middle_budget].astype(np.int64, copy=False)


def _select_propagation_indices(
    row: dict[str, Any],
    max_nodes: int,
    *,
    stochastic: bool,
    early_fraction: float,
    middle_fraction: float,
    sampling_seed: int | None = None,
    sampling_epoch: int = 0,
) -> np.ndarray:
    node_count = int(row["node_features_continuous"].shape[0])
    if node_count <= max_nodes:
        return np.arange(node_count, dtype=np.int64)
    raw_n_nodes = int(row.get("raw_n_nodes", node_count))
    if raw_n_nodes <= max_nodes:
        return np.arange(min(node_count, max_nodes), dtype=np.int64)
    role_ids = np.asarray(row.get("node_sampling_role_ids", np.zeros((node_count,), dtype=np.int16)))
    if role_ids.shape[0] != node_count or not np.any(role_ids == NODE_SAMPLING_ROLE_MIDDLE_POOL):
        return np.arange(min(node_count, max_nodes), dtype=np.int64)

    early_budget, middle_budget, late_budget = _budget_counts(max_nodes, early_fraction, middle_fraction)
    source_indices = np.flatnonzero(role_ids == NODE_SAMPLING_ROLE_SOURCE)[:1]
    early_indices = np.flatnonzero(role_ids == NODE_SAMPLING_ROLE_EARLY)[:early_budget]
    middle_pool_indices = np.flatnonzero(role_ids == NODE_SAMPLING_ROLE_MIDDLE_POOL)
    middle_indices = _middle_window_indices_seeded(
        middle_pool_indices,
        middle_budget,
        stochastic=stochastic,
        sampling_seed=sampling_seed,
        sampling_epoch=sampling_epoch,
    )
    late_role_indices = np.flatnonzero(role_ids == NODE_SAMPLING_ROLE_LATE)
    late_indices = late_role_indices[-late_budget:] if late_budget > 0 else np.zeros((0,), dtype=np.int64)
    keep = np.concatenate([source_indices, early_indices, middle_indices, late_indices]).astype(np.int64, copy=False)
    if keep.size == 0:
        return np.arange(min(node_count, max_nodes), dtype=np.int64)
    keep = np.unique(keep)
    if keep.size > max_nodes:
        keep = keep[:max_nodes]
    return keep


def _trim_propagation_arrays(
    row: dict[str, Any],
    max_nodes: int | None,
    *,
    stochastic: bool = False,
    early_fraction: float = 1.0 / 3.0,
    middle_fraction: float = 1.0 / 3.0,
    propagation_sampling_seed: int | None = None,
    propagation_sampling_epoch: int = 0,
    event_id: Any | None = None,
) -> dict[str, Any]:
    if max_nodes is None:
        return row
    node_count = row["node_features_continuous"].shape[0]
    if node_count <= max_nodes and int(row.get("raw_n_nodes", node_count)) <= max_nodes:
        return row
    keep_indices = _select_propagation_indices(
        row,
        int(max_nodes),
        stochastic=stochastic,
        early_fraction=early_fraction,
        middle_fraction=middle_fraction,
        sampling_seed=(int(propagation_sampling_seed) if stochastic and propagation_sampling_seed is not None and event_id is not None else None),
        sampling_epoch=propagation_sampling_epoch,
    )
    return _slice_propagation_arrays(row, keep_indices)


def _fill_stacked_tensor(rows: list[np.ndarray], *, dtype: torch.dtype) -> torch.Tensor:
    first = rows[0]
    shape = (len(rows), *first.shape)
    out = torch.empty(shape, dtype=dtype)
    for idx, row in enumerate(rows):
        out[idx].copy_(_numpy_to_torch(row))
    return out


def collate_fusion_batch(
    rows: list[dict[str, Any]],
    max_replies_per_event: int | None = None,
    max_propagation_nodes_per_event: int | None = None,
    stochastic_propagation_node_sampling: bool = False,
    propagation_early_fraction: float = 1.0 / 3.0,
    propagation_middle_fraction: float = 1.0 / 3.0,
    propagation_sampling_seed: int | None = None,
    propagation_sampling_epoch: int = 0,
) -> FusionBatch:
    if not rows:
        raise ValueError("collate_fusion_batch requires at least one row.")
    batch_size = len(rows)
    first_row = rows[0]

    content = {
        "input_ids": _fill_stacked_tensor([row["content"]["input_ids"] for row in rows], dtype=torch.long),
        "attention_mask": _fill_stacked_tensor([row["content"]["attention_mask"] for row in rows], dtype=torch.long),
        "surface_features": _fill_stacked_tensor([row["content"]["surface_features"] for row in rows], dtype=torch.float32),
        "quality_flags": _fill_stacked_tensor([row["content"]["quality_flags"] for row in rows], dtype=torch.float32),
    }

    first_stance_row = first_row["stance"]
    stance_reply_counts: list[int] = []
    stance_rows: list[dict[str, Any]] = []
    total_replies = 0
    for row in rows:
        stance_row = row["stance"]
        reply_count = stance_row["reply_embedding"].shape[0]
        if max_replies_per_event is not None and reply_count > max_replies_per_event:
            keep = slice(0, max_replies_per_event)
            stance_row = {
                key: value[keep] if getattr(value, "ndim", 0) > 0 else value
                for key, value in stance_row.items()
            }
            reply_count = stance_row["reply_embedding"].shape[0]
        stance_rows.append(stance_row)
        stance_reply_counts.append(reply_count)
        total_replies += reply_count

    embedding_dim = int(first_stance_row["reply_embedding"].shape[1]) if first_stance_row["reply_embedding"].ndim == 2 else 0
    stance_context_cat_dim = int(first_stance_row["context_cat"].shape[1]) if first_stance_row["context_cat"].ndim == 2 else 4
    stance_context_dense_dim = int(first_stance_row["context_dense"].shape[1]) if first_stance_row["context_dense"].ndim == 2 else 17
    reply_embedding = torch.empty((total_replies, embedding_dim), dtype=torch.float32)
    context_cat = torch.empty((total_replies, stance_context_cat_dim), dtype=torch.long)
    context_dense = torch.empty((total_replies, stance_context_dense_dim), dtype=torch.float32)
    reply_label_ids = torch.empty((total_replies,), dtype=torch.long)
    reply_label_mask = torch.empty((total_replies,), dtype=torch.float32)
    event_index = torch.empty((total_replies,), dtype=torch.long)
    event_sampled_reply_count = torch.empty((total_replies,), dtype=torch.float32)
    event_truncation_flag = torch.empty((total_replies,), dtype=torch.float32)
    cursor = 0
    for batch_idx, stance_row in enumerate(stance_rows):
        reply_count = stance_reply_counts[batch_idx]
        if reply_count == 0:
            continue
        next_cursor = cursor + reply_count
        reply_embedding[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["reply_embedding"], dtype=np.float32))
        context_cat[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["context_cat"], dtype=np.int64))
        context_dense[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["context_dense"], dtype=np.float32))
        reply_label_ids[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["reply_label_ids"], dtype=np.int64))
        reply_label_mask[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["reply_label_mask"], dtype=np.float32))
        event_index[cursor:next_cursor].fill_(batch_idx)
        event_sampled_reply_count[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["event_sampled_reply_count"], dtype=np.float32))
        event_truncation_flag[cursor:next_cursor].copy_(_numpy_to_torch(stance_row["event_truncation_flag"], dtype=np.float32))
        cursor = next_cursor
    stance = {
        "reply_embedding": reply_embedding,
        "context_cat": context_cat,
        "context_dense": context_dense,
        "reply_label_ids": reply_label_ids,
        "reply_label_mask": reply_label_mask,
        "event_index": event_index,
        "event_sampled_reply_count": event_sampled_reply_count,
        "event_truncation_flag": event_truncation_flag,
    }

    propagation_rows: list[dict[str, Any]] = []
    propagation_counts: list[int] = []
    total_nodes = 0
    for row in rows:
        propagation_row = _trim_propagation_arrays(
            row["propagation"],
            max_propagation_nodes_per_event,
            stochastic=stochastic_propagation_node_sampling,
            early_fraction=propagation_early_fraction,
            middle_fraction=propagation_middle_fraction,
            propagation_sampling_seed=propagation_sampling_seed,
            propagation_sampling_epoch=propagation_sampling_epoch,
            event_id=row.get("event_id"),
        )
        node_count = propagation_row["node_features_continuous"].shape[0]
        propagation_rows.append(propagation_row)
        propagation_counts.append(node_count)
        total_nodes += node_count
    prop_cont_dim = max((row["propagation"]["node_features_continuous"].shape[1] for row in rows if row["propagation"]["node_features_continuous"].ndim == 2), default=6)
    prop_cat_dim = max((row["propagation"]["node_features_categorical"].shape[1] for row in rows if row["propagation"]["node_features_categorical"].ndim == 2), default=11)
    prop_event_summary_dim = max((row["propagation"]["event_summary_features"].shape[0] for row in rows), default=0)
    prop_temporal_profile_dim = max((row["propagation"]["temporal_profile_features"].shape[0] for row in rows), default=0)
    node_features_continuous = torch.empty((total_nodes, prop_cont_dim), dtype=torch.float32)
    node_features_categorical = torch.empty((total_nodes, prop_cat_dim), dtype=torch.long)
    node_masks = torch.empty((total_nodes,), dtype=torch.float32)
    parent_index = torch.empty((total_nodes,), dtype=torch.long)
    prop_event_index = torch.empty((total_nodes,), dtype=torch.long)
    event_summary_features = torch.zeros((batch_size, prop_event_summary_dim), dtype=torch.float32)
    temporal_profile_features = torch.zeros((batch_size, prop_temporal_profile_dim), dtype=torch.float32)
    cursor = 0
    for batch_idx, propagation_row in enumerate(propagation_rows):
        node_count = propagation_counts[batch_idx]
        if prop_event_summary_dim > 0:
            current = _numpy_to_torch(propagation_row["event_summary_features"], dtype=np.float32)
            event_summary_features[batch_idx, : current.shape[0]].copy_(current)
        if prop_temporal_profile_dim > 0:
            current = _numpy_to_torch(propagation_row["temporal_profile_features"], dtype=np.float32)
            temporal_profile_features[batch_idx, : current.shape[0]].copy_(current)
        if node_count == 0:
            continue
        next_cursor = cursor + node_count
        node_features_continuous[cursor:next_cursor].copy_(_numpy_to_torch(propagation_row["node_features_continuous"], dtype=np.float32))
        node_features_categorical[cursor:next_cursor].copy_(_numpy_to_torch(propagation_row["node_features_categorical"], dtype=np.int64))
        node_masks[cursor:next_cursor].copy_(_numpy_to_torch(propagation_row["node_masks"], dtype=np.float32))
        parent_index[cursor:next_cursor].copy_(_numpy_to_torch(propagation_row["parent_index"], dtype=np.int64))
        prop_event_index[cursor:next_cursor].fill_(batch_idx)
        cursor = next_cursor
    propagation = {
        "node_features_continuous": node_features_continuous,
        "node_features_categorical": node_features_categorical,
        "node_masks": node_masks,
        "parent_index": parent_index,
        "event_index": prop_event_index,
        "event_summary_features": event_summary_features,
        "temporal_profile_features": temporal_profile_features,
    }

    evidence_summary = _fill_stacked_tensor([row["evidence"]["summary_features"] for row in rows], dtype=torch.float32)
    evidence_quality = _fill_stacked_tensor([row["evidence"]["summary_quality_flags"] for row in rows], dtype=torch.float32)
    evidence_ref_dim = max((row["evidence"]["ref_row_features"].shape[1] for row in rows if row["evidence"]["ref_row_features"].ndim == 2), default=0)
    evidence_max_refs = max((row["evidence"]["ref_row_features"].shape[0] for row in rows), default=0)
    evidence_ref_row_features = torch.zeros((batch_size, evidence_max_refs, evidence_ref_dim), dtype=torch.float32)
    evidence_ref_row_mask = torch.zeros((batch_size, evidence_max_refs), dtype=torch.float32)
    for batch_idx, row in enumerate(rows):
        ref_features = row["evidence"]["ref_row_features"]
        ref_mask = row["evidence"]["ref_row_mask"]
        ref_count = int(ref_features.shape[0])
        if ref_count == 0:
            continue
        evidence_ref_row_features[batch_idx, :ref_count].copy_(_numpy_to_torch(ref_features, dtype=np.float32))
        evidence_ref_row_mask[batch_idx, :ref_count].copy_(_numpy_to_torch(ref_mask[:ref_count], dtype=np.float32))
    evidence = {
        "summary_features": evidence_summary,
        "summary_quality_flags": evidence_quality,
        "ref_row_features": evidence_ref_row_features,
        "ref_row_mask": evidence_ref_row_mask,
    }

    source = {
        "summary_features": _fill_stacked_tensor([row["source"]["summary_features"] for row in rows], dtype=torch.float32),
        "summary_quality_flags": _fill_stacked_tensor([row["source"]["summary_quality_flags"] for row in rows], dtype=torch.float32),
        "categorical_features": _fill_stacked_tensor([row["source"]["categorical_features"] for row in rows], dtype=torch.long),
    }

    modality_mask = {
        view: torch.tensor([row["modality_mask"][view] for row in rows], dtype=torch.float32)
        for view in ["content", "stance", "propagation", "evidence", "source"]
    }
    primary_label = torch.tensor([row["labels"]["primary_task"] for row in rows], dtype=torch.long)

    return FusionBatch(
        event_ids=[row["event_id"] for row in rows],
        dataset_ids=[row["dataset_id"] for row in rows],
        platforms=[row["platform"] for row in rows],
        canonical_splits=[row["canonical_split"] for row in rows],
        raw_split_sources=[row["raw_split_source"] for row in rows],
        modalities={
            "content": content,
            "stance": stance,
            "propagation": propagation,
            "evidence": evidence,
            "source": source,
        },
        modality_mask=modality_mask,
        task_labels={
            "primary_task": primary_label,
        },
        task_mask={
            "primary_task": (primary_label >= 0).to(torch.float32),
        },
        metadata=[row["metadata"] for row in rows],
    )


class FusionBatchCollator:
    def __init__(
        self,
        *,
        max_replies_per_event: int | None = None,
        max_propagation_nodes_per_event: int | None = None,
        stochastic_propagation_node_sampling: bool = False,
        propagation_early_fraction: float = 1.0 / 3.0,
        propagation_middle_fraction: float = 1.0 / 3.0,
        propagation_sampling_seed: int | None = None,
    ) -> None:
        self.max_replies_per_event = max_replies_per_event
        self.max_propagation_nodes_per_event = max_propagation_nodes_per_event
        self.stochastic_propagation_node_sampling = bool(stochastic_propagation_node_sampling)
        self.propagation_early_fraction = float(propagation_early_fraction)
        self.propagation_middle_fraction = float(propagation_middle_fraction)
        self.propagation_sampling_seed = propagation_sampling_seed
        self._epoch = multiprocessing.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def get_epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    def __call__(self, rows: list[dict[str, Any]]) -> FusionBatch:
        return collate_fusion_batch(
            rows,
            max_replies_per_event=self.max_replies_per_event,
            max_propagation_nodes_per_event=self.max_propagation_nodes_per_event,
            stochastic_propagation_node_sampling=self.stochastic_propagation_node_sampling,
            propagation_early_fraction=self.propagation_early_fraction,
            propagation_middle_fraction=self.propagation_middle_fraction,
            propagation_sampling_seed=self.propagation_sampling_seed,
            propagation_sampling_epoch=self.get_epoch(),
        )
