from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config.cli import add_resolver_args
from config.resolver import resolve_encoder_config
from models.encoders.propagation_view.array_writer import write_propagation_manifest, write_propagation_shard
from models.encoders.propagation_view.config import Mode, PropagationViewConfig
from models.encoders.propagation_view.dataset import iter_preprocessed_events
from models.encoders.propagation_view.label_alignment import MISSING_LABEL_ID, align_propagation_event_labels
from models.encoders.propagation_view.node_selector import propagation_mode_for_nodes, select_and_order_real_nodes
from models.encoders.propagation_view.schemas import PropagationArrayShard, PropagationEventMeta, PropagationNodeExampleMeta
from models.encoders.propagation_view.state_builder import interaction_type_id, is_boilerplate_text, text_status_id
from models.encoders.propagation_view.structure_builder import depth_bucket_id
from models.encoders.propagation_view.time_builder import continuous_time_values, observation_kind_id, time_bucket_id, time_sort_key
from utils.preprocessed_store import PREPROCESSED_EVENTS_FILENAME


ACTIVE_CONTINUOUS_FEATURE_NAMES = [
    "time_from_source_min_value",
    "log1p_time_from_source_min_value",
    "has_continuous_time_value",
    "depth_value",
    "log1p_child_count_value",
    "parent_child_time_gap_min_value",
    "normalized_node_rank_value",
    "local_branching_change_value",
    "super_node_size_value",
    "time_span_min_value",
]

ACTIVE_CATEGORICAL_FEATURE_NAMES = [
    "depth_bucket_id",
    "time_bucket_id",
    "observation_kind_id",
    "interaction_type_id",
    "text_status_id",
    "parent_is_source",
    "is_placeholder",
    "is_observed_text",
    "order_bucket_id",
    "parent_inferred",
    "is_leaf",
    "sibling_count_bucket_id",
    "subtree_size_bucket_id",
    "stage_bucket_id",
    "is_super_node",
    "super_node_size_bucket_id",
]

EVENT_SUMMARY_FEATURE_NAMES = [
    "n_nodes_value",
    "n_replies_value",
    "max_depth_value",
    "mean_depth_value",
    "leaf_ratio_value",
    "reply_ratio_value",
    "repost_ratio_value",
    "quote_repost_ratio_value",
    "placeholder_ratio_value",
    "boilerplate_ratio_value",
    "text_observed_ratio_value",
    "mean_child_count_value",
    "max_width_value",
    "early_node_ratio_value",
    "mid_node_ratio_value",
    "late_node_ratio_value",
    "time_to_25pct_nodes_value",
    "time_to_50pct_nodes_value",
    "time_to_75pct_nodes_value",
    "burst_density_value",
    "depth1_ratio_value",
    "depth2_ratio_value",
    "depth3plus_ratio_value",
]

TEMPORAL_PROFILE_FEATURE_NAMES = [
    "coarse_count_unknown",
    "coarse_count_early",
    "coarse_count_mid",
    "coarse_count_late",
    "coarse_ratio_unknown",
    "coarse_ratio_early",
    "coarse_ratio_mid",
    "coarse_ratio_late",
    "fine_count_unknown",
    "fine_count_0_5min",
    "fine_count_5_30min",
    "fine_count_30min_2h",
    "fine_count_2h_6h",
    "fine_count_6h_24h",
    "fine_count_gt_24h",
    "reply_count_early",
    "reply_count_mid",
    "reply_count_late",
    "repost_count_early",
    "repost_count_mid",
    "repost_count_late",
    "quote_repost_count_early",
    "quote_repost_count_mid",
    "quote_repost_count_late",
    "depth1_count_early",
    "depth2_count_early",
    "depth3plus_count_early",
    "depth1_count_mid",
    "depth2_count_mid",
    "depth3plus_count_mid",
    "depth1_count_late",
    "depth2_count_late",
    "depth3plus_count_late",
    "cumulative_ratio_le_5min",
    "cumulative_ratio_le_30min",
    "cumulative_ratio_le_2h",
    "cumulative_ratio_le_6h",
    "cumulative_ratio_le_24h",
]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _node_count_bucket_id(value: int, edges: list[int]) -> int:
    for idx, edge in enumerate(edges):
        if int(value) <= int(edge):
            return idx
    return len(edges)


def _resolve_bucket_edges(event_meta_rows: list[dict[str, Any]]) -> tuple[list[int], str]:
    if not event_meta_rows:
        return [0, 0, 0, 0], "all"
    train_values = [int(row["node_count_value"]) for row in event_meta_rows if str(row.get("canonical_split") or "").strip().lower() == "train"]
    source_split = "train"
    if not train_values:
        train_values = [int(row["node_count_value"]) for row in event_meta_rows]
        source_split = "all"
    quantiles = np.quantile(np.asarray(train_values, dtype=np.float32), [0.33, 0.66], method="linear")
    edges = [int(round(float(value))) for value in quantiles.tolist()]
    return edges, source_split


def _log1p_child_count(value: int | float) -> float:
    return math.log1p(max(0.0, float(value)))


def _prepare_nodes(raw_nodes: list[dict]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in raw_nodes:
        cloned = dict(node)
        cloned["_time_sort_key"] = time_sort_key(cloned)
        cloned["_is_super_node"] = False
        nodes.append(cloned)
    return nodes


def _coarse_stage_bucket(node_time: dict[str, Any]) -> int:
    return time_bucket_id(node_time)


def _fine_time_bucket_id(node_time: dict[str, Any]) -> int:
    minutes = node_time.get("time_from_source_min")
    if minutes is None:
        values = node_time.get("time_delay_minutes_raw_values") or []
        if values:
            try:
                minutes = float(values[0])
            except (TypeError, ValueError):
                minutes = None
    if minutes is None:
        return 0
    if minutes <= 5:
        return 1
    if minutes <= 30:
        return 2
    if minutes <= 120:
        return 3
    if minutes <= 360:
        return 4
    if minutes <= 1440:
        return 5
    return 6


def _bucket_sibling_count(value: int) -> int:
    if value <= 0:
        return 1
    if value <= 2:
        return 2
    if value <= 5:
        return 3
    return 4


def _bucket_subtree_size(value: int) -> int:
    if value <= 1:
        return 1
    if value <= 4:
        return 2
    if value <= 12:
        return 3
    return 4


def _bucket_super_node_size(value: int) -> int:
    if value <= 1:
        return 1
    if value <= 4:
        return 2
    if value <= 16:
        return 3
    return 4


def _compute_tree_statistics(nodes: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, float]]:
    child_counts: dict[str, int] = Counter()
    sibling_counts: dict[str, int] = {}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)
    node_map = {str(node["post_id"]): node for node in nodes}
    for node in nodes:
        parent_post_id = node.get("parent_post_id")
        if parent_post_id:
            child_counts[str(parent_post_id)] += 1
            children_by_parent[str(parent_post_id)].append(str(node["post_id"]))
    for node in nodes:
        parent_post_id = node.get("parent_post_id")
        if parent_post_id:
            sibling_counts[str(node["post_id"])] = max(0, len(children_by_parent[str(parent_post_id)]) - 1)
        else:
            sibling_counts[str(node["post_id"])] = 0

    subtree_sizes: dict[str, int] = {}

    def _dfs(post_id: str) -> int:
        total = 1
        for child_id in children_by_parent.get(post_id, []):
            total += _dfs(child_id)
        subtree_sizes[post_id] = total
        return total

    roots = [str(node["post_id"]) for node in nodes if node.get("is_source") or not node.get("parent_post_id")]
    for root_id in roots:
        if root_id in node_map:
            _dfs(root_id)
    widths: dict[int, int] = Counter()
    for node in nodes:
        widths[int(node.get("depth") or 0)] += 1
    width_stats = {
        "max_width": float(max(widths.values()) if widths else 0.0),
    }
    return dict(child_counts), sibling_counts, subtree_sizes, width_stats


def _make_super_node(group_key: tuple[Any, ...], grouped_nodes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    time_bucket, depth_bucket, interaction_type, text_status, parent_is_source, observation_kind = group_key
    minutes = [float((node.get("time") or {}).get("time_from_source_min")) for node in grouped_nodes if (node.get("time") or {}).get("time_from_source_min") is not None]
    mean_minutes = float(sum(minutes) / len(minutes)) if minutes else 0.0
    span_minutes = float(max(minutes) - min(minutes)) if len(minutes) >= 2 else 0.0
    depths = [float(node.get("depth") or 0.0) for node in grouped_nodes]
    mean_depth = float(sum(depths) / len(depths)) if depths else 0.0
    child_counts = [int(node.get("_child_count", 0)) for node in grouped_nodes]
    placeholder_ratio = _safe_div(sum(1 for node in grouped_nodes if node.get("is_placeholder")), len(grouped_nodes))
    boilerplate_ratio = _safe_div(sum(1 for node in grouped_nodes if is_boilerplate_text(node.get("text_status"))), len(grouped_nodes))
    leaf_ratio = _safe_div(sum(1 for value in child_counts if value == 0), len(grouped_nodes))
    mean_child_count = float(sum(child_counts) / len(child_counts)) if child_counts else 0.0
    return {
        "post_id": f"super_node_{index}",
        "parent_post_id": None,
        "parent_is_source": bool(parent_is_source),
        "parent_inferred": True,
        "is_source": False,
        "interaction_type": "reply" if interaction_type == 2 else ("repost" if interaction_type == 3 else ("quote_repost" if interaction_type == 4 else "unknown")),
        "text_status": "boilerplate" if text_status == 2 else ("placeholder" if text_status == 4 else ("substantive" if text_status == 1 else "unknown")),
        "is_placeholder": placeholder_ratio >= 0.5,
        "is_observed_text": any(node.get("is_observed_text") for node in grouped_nodes),
        "depth": int(round(mean_depth)),
        "time": {
            "observation_kind": "relative_delay_observed" if minutes else "missing",
            "time_from_source_min": mean_minutes if minutes else None,
            "time_from_source_sec": mean_minutes * 60.0 if minutes else None,
        },
        "_time_sort_key": (1 if minutes else 2, mean_minutes if minutes else float(index), index, f"super_node_{index}"),
        "_keep_reason": "super_node",
        "_is_super_node": True,
        "_super_node_size": len(grouped_nodes),
        "_time_span_min": span_minutes,
        "_super_mean_depth": mean_depth,
        "_super_mean_child_count": mean_child_count,
        "_super_leaf_ratio": leaf_ratio,
        "_super_placeholder_ratio": placeholder_ratio,
        "_super_boilerplate_ratio": boilerplate_ratio,
        "_grouped_nodes": grouped_nodes,
        "_forced_time_bucket_id": int(time_bucket),
        "_forced_depth_bucket_id": int(depth_bucket),
        "_forced_observation_kind_id": int(observation_kind),
        "_forced_interaction_type_id": int(interaction_type),
        "_forced_text_status_id": int(text_status),
        "_forced_stage_bucket_id": int(time_bucket),
    }


def _select_nodes_with_super_nodes(raw_nodes: list[dict[str, Any]], config: PropagationViewConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_n_nodes = len(raw_nodes)
    selected_real_nodes, selection_stats = select_and_order_real_nodes(raw_nodes, config)
    return selected_real_nodes, {
        "raw_n_nodes": raw_n_nodes,
        "selected_real_nodes": len(selected_real_nodes),
        "selected_super_nodes": 0,
        "compression_triggered": False,
        "n_super_nodes": 0,
        "dropped_by_truncation": selection_stats["dropped_by_truncation"],
    }


def _quantile_minutes(sorted_minutes: list[float], ratio: float) -> float:
    if not sorted_minutes:
        return 0.0
    index = min(len(sorted_minutes) - 1, max(0, int(math.ceil(len(sorted_minutes) * ratio) - 1)))
    return float(sorted_minutes[index])


def _build_event_level_features(nodes: list[dict[str, Any]], child_counts: dict[str, int], width_stats: dict[str, float]) -> tuple[list[float], list[float]]:
    n_nodes = len(nodes)
    if n_nodes == 0:
        return [0.0 for _ in EVENT_SUMMARY_FEATURE_NAMES], [0.0 for _ in TEMPORAL_PROFILE_FEATURE_NAMES]
    depths = [float(node.get("depth") or 0.0) for node in nodes]
    interaction_counter = Counter(node.get("interaction_type") or "unknown" for node in nodes)
    placeholder_count = sum(1 for node in nodes if node.get("is_placeholder"))
    boilerplate_count = sum(1 for node in nodes if is_boilerplate_text(node.get("text_status")))
    observed_text_count = sum(1 for node in nodes if node.get("is_observed_text"))
    leaf_count = sum(1 for node in nodes if child_counts.get(str(node["post_id"]), 0) == 0)
    stage_counts = Counter(_coarse_stage_bucket(node.get("time") or {}) for node in nodes)
    depth1_count = sum(1 for node in nodes if (node.get("depth") or 0) == 1)
    depth2_count = sum(1 for node in nodes if (node.get("depth") or 0) == 2)
    depth3plus_count = sum(1 for node in nodes if (node.get("depth") or 0) >= 3)
    time_minutes = sorted(
        float((node.get("time") or {}).get("time_from_source_min"))
        for node in nodes
        if (node.get("time") or {}).get("time_from_source_min") is not None
    )
    burst_density = _safe_div(sum(1 for value in time_minutes if value <= 30.0), max(1, n_nodes))
    event_summary = [
        float(n_nodes),
        float(interaction_counter.get("reply", 0)),
        float(max(depths) if depths else 0.0),
        float(sum(depths) / len(depths)) if depths else 0.0,
        _safe_div(leaf_count, n_nodes),
        _safe_div(interaction_counter.get("reply", 0), n_nodes),
        _safe_div(interaction_counter.get("repost", 0), n_nodes),
        _safe_div(interaction_counter.get("quote_repost", 0), n_nodes),
        _safe_div(placeholder_count, n_nodes),
        _safe_div(boilerplate_count, n_nodes),
        _safe_div(observed_text_count, n_nodes),
        float(sum(child_counts.values()) / max(1, len(child_counts))) if child_counts else 0.0,
        float(width_stats.get("max_width", 0.0)),
        _safe_div(stage_counts.get(1, 0), n_nodes),
        _safe_div(stage_counts.get(2, 0), n_nodes),
        _safe_div(stage_counts.get(3, 0), n_nodes),
        _quantile_minutes(time_minutes, 0.25),
        _quantile_minutes(time_minutes, 0.50),
        _quantile_minutes(time_minutes, 0.75),
        burst_density,
        _safe_div(depth1_count, n_nodes),
        _safe_div(depth2_count, n_nodes),
        _safe_div(depth3plus_count, n_nodes),
    ]

    coarse_counts = [stage_counts.get(index, 0) for index in range(4)]
    fine_counts = Counter(_fine_time_bucket_id(node.get("time") or {}) for node in nodes)
    reply_stage_counts = [0, 0, 0]
    repost_stage_counts = [0, 0, 0]
    quote_stage_counts = [0, 0, 0]
    depth_stage_counts = [[0, 0, 0] for _ in range(3)]
    for node in nodes:
        stage_id = _coarse_stage_bucket(node.get("time") or {})
        if stage_id not in {1, 2, 3}:
            continue
        stage_index = stage_id - 1
        interaction = node.get("interaction_type") or "unknown"
        if interaction == "reply":
            reply_stage_counts[stage_index] += 1
        elif interaction == "repost":
            repost_stage_counts[stage_index] += 1
        elif interaction == "quote_repost":
            quote_stage_counts[stage_index] += 1
        depth_value = int(node.get("depth") or 0)
        if depth_value == 1:
            depth_stage_counts[stage_index][0] += 1
        elif depth_value == 2:
            depth_stage_counts[stage_index][1] += 1
        elif depth_value >= 3:
            depth_stage_counts[stage_index][2] += 1

    thresholds = [5.0, 30.0, 120.0, 360.0, 1440.0]
    cumulative_ratios = []
    for threshold in thresholds:
        cumulative_ratios.append(_safe_div(sum(1 for value in time_minutes if value <= threshold), n_nodes))

    temporal_profile = [
        *[float(value) for value in coarse_counts],
        *[_safe_div(value, n_nodes) for value in coarse_counts],
        *[float(fine_counts.get(index, 0)) for index in range(7)],
        *[float(value) for value in reply_stage_counts],
        *[float(value) for value in repost_stage_counts],
        *[float(value) for value in quote_stage_counts],
        *[float(value) for group in depth_stage_counts for value in group],
        *[float(value) for value in cumulative_ratios],
    ]
    return event_summary, temporal_profile


def _count_time_observation_stats(nodes: list[dict[str, Any]]) -> tuple[int, int, int]:
    observed_timestamps = 0
    relative_delay_nodes = 0
    missing_time_nodes = 0
    for node in nodes:
        kind = (node.get("time") or {}).get("observation_kind") or "missing"
        if kind != "missing":
            observed_timestamps += 1
        if kind == "relative_delay_observed":
            relative_delay_nodes += 1
        if kind == "missing":
            missing_time_nodes += 1
    return observed_timestamps, relative_delay_nodes, missing_time_nodes


def _build_node_feature_rows(nodes: list[dict[str, Any]], config: PropagationViewConfig) -> tuple[list[list[float]], list[list[int]], list[int], list[dict[str, Any]], dict[str, int]]:
    child_counts, sibling_counts, subtree_sizes, width_stats = _compute_tree_statistics(nodes)
    for node in nodes:
        node["_child_count"] = child_counts.get(str(node["post_id"]), 0)
    root_post_id = next((str(node["post_id"]) for node in nodes if node.get("is_source")), None)
    node_index_map = {str(node["post_id"]): idx for idx, node in enumerate(nodes)}
    continuous_rows: list[list[float]] = []
    categorical_rows: list[list[int]] = []
    parent_indices: list[int] = []
    node_meta_rows: list[dict[str, Any]] = []

    for node_order_index, node in enumerate(nodes):
        node_time = node.get("time") or {}
        kind = node_time.get("observation_kind") or "missing"
        minute_value = node_time.get("time_from_source_min")
        sec_value, minute_value_fallback, has_continuous = continuous_time_values(node_time)
        minute_value = float(minute_value if minute_value is not None else minute_value_fallback)
        depth_value = float(node.get("depth") or 0.0)
        parent_post_id = str(node.get("parent_post_id")) if node.get("parent_post_id") is not None else None
        if node.get("_is_super_node"):
            parent_idx = node_index_map.get(root_post_id, -1) if node.get("parent_is_source") and root_post_id is not None else -1
        else:
            parent_idx = -1 if parent_post_id is None else node_index_map.get(parent_post_id, -1)
        parent_indices.append(parent_idx)
        parent_gap_min = 0.0
        local_branching_change = _log1p_child_count(node.get("_child_count", 0))
        if parent_idx >= 0:
            parent_node = nodes[parent_idx]
            parent_time = parent_node.get("time") or {}
            parent_min = parent_time.get("time_from_source_min")
            if parent_min is not None and has_continuous:
                parent_gap_min = max(0.0, minute_value - float(parent_min))
            local_branching_change -= _log1p_child_count(parent_node.get("_child_count", 0))
        normalized_rank = float(node_order_index) / float(max(1, len(nodes) - 1))
        super_node_size = float(node.get("_super_node_size", 1 if node.get("_is_super_node") else 0))
        time_span_min = float(node.get("_time_span_min", 0.0))
        continuous_rows.append(
            [
                minute_value,
                math.log1p(max(0.0, minute_value)),
                float(has_continuous),
                depth_value,
                _log1p_child_count(node.get("_child_count", 0)),
                parent_gap_min,
                normalized_rank,
                local_branching_change,
                super_node_size,
                time_span_min,
            ]
        )
        categorical_rows.append(
            [
                int(node.get("_forced_depth_bucket_id", depth_bucket_id(node.get("depth")))),
                int(node.get("_forced_time_bucket_id", time_bucket_id(node_time))),
                int(node.get("_forced_observation_kind_id", observation_kind_id(kind))),
                int(node.get("_forced_interaction_type_id", interaction_type_id(node.get("interaction_type")))),
                int(node.get("_forced_text_status_id", text_status_id(node.get("text_status")))),
                1 if node.get("parent_is_source") else 0,
                1 if node.get("is_placeholder") else 0,
                1 if node.get("is_observed_text") else 0,
                1 if node_order_index <= 2 else (2 if node_order_index <= 8 else 3),
                1 if node.get("parent_inferred") else 0,
                1 if node.get("_child_count", 0) == 0 else 0,
                _bucket_sibling_count(sibling_counts.get(str(node["post_id"]), 0)),
                _bucket_subtree_size(subtree_sizes.get(str(node["post_id"]), 1)),
                int(node.get("_forced_stage_bucket_id", _coarse_stage_bucket(node_time))),
                1 if node.get("_is_super_node") else 0,
                _bucket_super_node_size(int(node.get("_super_node_size", 0))),
            ]
        )
        meta = PropagationNodeExampleMeta(
            event_id="",
            dataset_id="",
            platform="",
            node_post_id=str(node["post_id"]),
            parent_post_id=parent_post_id,
            parent_index=parent_idx,
            is_source=bool(node.get("is_source")),
            is_super_node=bool(node.get("_is_super_node")),
            node_order_index=node_order_index,
            time_observation_kind=kind,
            time_observation_kind_id=observation_kind_id(kind),
            created_at=node_time.get("created_at"),
            time_from_source_sec=node_time.get("time_from_source_sec"),
            time_from_source_min=node_time.get("time_from_source_min"),
            time_bucket_id=time_bucket_id(node_time),
            stage_bucket_id=_coarse_stage_bucket(node_time),
            depth_value=(int(node.get("depth")) if node.get("depth") is not None else None),
            depth_bucket_id=depth_bucket_id(node.get("depth")),
            root_distance=(int(node.get("depth")) if node.get("depth") is not None else None),
            parent_is_source=bool(node.get("parent_is_source")),
            parent_inferred=bool(node.get("parent_inferred")),
            interaction_type=node.get("interaction_type") or "unknown",
            interaction_type_id=interaction_type_id(node.get("interaction_type")),
            text_status=node.get("text_status") or "unknown",
            text_status_id=text_status_id(node.get("text_status")),
            is_placeholder=bool(node.get("is_placeholder")),
            is_observed_text=bool(node.get("is_observed_text")),
            is_boilerplate_text=bool(is_boilerplate_text(node.get("text_status"))),
            has_continuous_time=bool(has_continuous),
            log1p_child_count=_log1p_child_count(node.get("_child_count", 0)),
            is_leaf=node.get("_child_count", 0) == 0,
            order_bucket_id=(1 if node_order_index <= 2 else (2 if node_order_index <= 8 else 3)),
            notes=node.get("_keep_reason", ""),
        )
        node_meta_rows.append(meta.__dict__.copy())

    return continuous_rows, categorical_rows, parent_indices, node_meta_rows, child_counts


def build_propagation_arrays(
    dataset_id: str,
    mode: Mode,
    input_root: Path,
    output_root: Path,
    *,
    config: PropagationViewConfig | None = None,
) -> None:
    config = config or resolve_encoder_config("propagation", dataset_id, mode)
    artifact_mode = config.artifact_mode(mode)
    data_root = input_root.parent
    input_path = input_root / dataset_id / PREPROCESSED_EVENTS_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed events file: {input_path}")
    dataset_dir = output_root / dataset_id / artifact_mode
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shard_id = 0
    shard_event_ids: list[str] = []
    shard_node_post_ids: list[str] = []
    shard_continuous: list[list[float]] = []
    shard_categorical: list[list[int]] = []
    shard_node_masks: list[int] = []
    shard_parent_index: list[int] = []
    shard_event_index: list[int] = []
    shard_label_ids: list[int] = []
    shard_label_mask: list[int] = []
    shard_event_summary: list[list[float]] = []
    shard_temporal_profile: list[list[float]] = []

    event_meta_rows: list[dict[str, Any]] = []
    node_meta_sample: list[dict[str, Any]] = []
    total_nodes = 0
    max_nodes_per_event = 0
    shard_filenames: list[str] = []
    truncation_drops = 0
    primary_class_names: list[str] = []

    def flush_shard() -> None:
        nonlocal shard_id
        if not shard_node_post_ids:
            return
        filename = f"propagation_shard_{artifact_mode}_{shard_id:04d}.npz"
        shard = PropagationArrayShard(
            shard_id=shard_id,
            event_ids=np.asarray(shard_event_ids, dtype=object),
            node_post_ids=np.asarray(shard_node_post_ids, dtype=object),
            node_features_continuous=np.asarray(shard_continuous, dtype=np.float32),
            node_features_categorical=np.asarray(shard_categorical, dtype=np.int16),
            node_masks=np.asarray(shard_node_masks, dtype=np.uint8),
            parent_index=np.asarray(shard_parent_index, dtype=np.int32),
            event_index=np.asarray(shard_event_index, dtype=np.int32),
            label_ids=np.asarray(shard_label_ids, dtype=np.int16),
            label_mask=np.asarray(shard_label_mask, dtype=np.uint8),
            event_summary_features=np.asarray(shard_event_summary, dtype=np.float32),
            temporal_profile_features=np.asarray(shard_temporal_profile, dtype=np.float32),
            max_nodes_per_event=max_nodes_per_event,
            n_node_examples=len(shard_node_post_ids),
        )
        write_propagation_shard(dataset_dir / filename, shard)
        shard_filenames.append(filename)
        shard_event_ids.clear()
        shard_node_post_ids.clear()
        shard_continuous.clear()
        shard_categorical.clear()
        shard_node_masks.clear()
        shard_parent_index.clear()
        shard_event_index.clear()
        shard_label_ids.clear()
        shard_label_mask.clear()
        shard_event_summary.clear()
        shard_temporal_profile.clear()
        shard_id += 1

    for event_index, event_row in enumerate(iter_preprocessed_events(input_path)):
        if shard_node_post_ids and len(shard_node_post_ids) >= config.shard_size:
            flush_shard()
        prop = event_row.get("propagation") or {}
        raw_nodes = _prepare_nodes(prop.get("nodes") or [])
        selected_nodes, selection_stats = _select_nodes_with_super_nodes(raw_nodes, config)
        truncation_drops += int(selection_stats["dropped_by_truncation"])
        continuous_rows, categorical_rows, parent_indices, node_meta_rows, _selected_child_counts = _build_node_feature_rows(selected_nodes, config)
        raw_child_counts, _raw_sibling_counts, _raw_subtree_sizes, raw_width_stats = _compute_tree_statistics(raw_nodes)
        event_summary_features, temporal_profile_features = _build_event_level_features(raw_nodes, raw_child_counts, raw_width_stats)
        labels = align_propagation_event_labels(dataset_id, event_row.get("task_targets") or {}, data_root=str(data_root))
        if not primary_class_names:
            primary_class_names = list(labels["primary_class_names"])
        label_id = int(labels["primary_label_id"])
        observed_timestamps, relative_delay_nodes, missing_time_nodes = _count_time_observation_stats(raw_nodes)

        node_start_index = total_nodes
        max_nodes_per_event = max(max_nodes_per_event, len(selected_nodes))
        event_shard_id = shard_id
        node_start_in_shard = len(shard_node_post_ids)

        for node_order_index, node in enumerate(selected_nodes):
            node_meta_rows[node_order_index]["event_id"] = event_row["event_id"]
            node_meta_rows[node_order_index]["dataset_id"] = dataset_id
            node_meta_rows[node_order_index]["platform"] = event_row["platform"]
            if len(node_meta_sample) < config.audit_sample_size:
                node_meta_sample.append(node_meta_rows[node_order_index].copy())
            shard_event_ids.append(event_row["event_id"])
            shard_node_post_ids.append(str(node["post_id"]))
            shard_continuous.append(continuous_rows[node_order_index])
            shard_categorical.append(categorical_rows[node_order_index])
            shard_node_masks.append(1)
            shard_parent_index.append(parent_indices[node_order_index])
            shard_event_index.append(event_index)
            shard_label_ids.append(label_id if label_id != MISSING_LABEL_ID else -1)
            shard_label_mask.append(0 if label_id == MISSING_LABEL_ID else 1)
            shard_event_summary.append(event_summary_features)
            shard_temporal_profile.append(temporal_profile_features)
            total_nodes += 1
        node_end_in_shard = len(shard_node_post_ids)

        event_meta = PropagationEventMeta(
            event_id=event_row["event_id"],
            dataset_id=dataset_id,
            platform=event_row["platform"],
            canonical_split=event_row["canonical_split"],
            raw_split_source=event_row.get("raw_split_source"),
            primary_task_name=str(labels["primary_task_name"]),
            primary_metric_name=str(labels["primary_metric_name"]),
            primary_label_name=labels["primary_label_name"],
            primary_label_id=label_id if label_id != MISSING_LABEL_ID else -1,
            propagation_mode=propagation_mode_for_nodes(raw_nodes),
            raw_n_nodes=int(selection_stats["raw_n_nodes"]),
            n_candidate_nodes=len(raw_nodes),
            n_encoded_nodes=len(selected_nodes),
            node_count_value=len(selected_nodes),
            node_count_bucket_id=0,
            n_observed_timestamps=observed_timestamps,
            n_relative_delay_only_nodes=relative_delay_nodes,
            n_missing_time_nodes=missing_time_nodes,
            selected_real_nodes=int(selection_stats["selected_real_nodes"]),
            selected_super_nodes=int(selection_stats["selected_super_nodes"]),
            compression_triggered=bool(selection_stats["compression_triggered"]),
            n_super_nodes=int(selection_stats["n_super_nodes"]),
            node_start_index=node_start_index,
            node_end_index=total_nodes,
            shard_id=event_shard_id,
            node_start_in_shard=node_start_in_shard,
            node_end_in_shard=node_end_in_shard,
            notes="summary_source=raw_nodes",
        )
        event_meta_rows.append(event_meta.__dict__.copy())

    flush_shard()
    bucket_quantile_edges, bucket_count_source_split = _resolve_bucket_edges(event_meta_rows)
    for row in event_meta_rows:
        row["node_count_bucket_id"] = _node_count_bucket_id(int(row["node_count_value"]), bucket_quantile_edges)

    manifest = {
        "dataset_id": dataset_id,
        "mode": artifact_mode,
        "artifact_family": config.resolved_artifact_family(),
        "encoder_family": config.resolved_encoder_family(),
        "shard_count": len(shard_filenames),
        "shard_filenames": shard_filenames,
        "total_event_count": len(event_meta_rows),
        "total_node_example_count": total_nodes,
        "max_nodes_per_event": max_nodes_per_event,
        "config_snapshot": config.as_dict(),
        "active_continuous_feature_names": ACTIVE_CONTINUOUS_FEATURE_NAMES,
        "active_categorical_feature_names": ACTIVE_CATEGORICAL_FEATURE_NAMES,
        "event_summary_feature_names": EVENT_SUMMARY_FEATURE_NAMES,
        "temporal_profile_feature_names": TEMPORAL_PROFILE_FEATURE_NAMES,
        "summary_source": "raw_nodes",
        "bucketing_policy": "train_quantile_node_count_5bucket",
        "bucket_quantile_edges": bucket_quantile_edges,
        "bucket_count": 5,
        "bucket_count_source_split": bucket_count_source_split,
        "pooling_policy": config.pooling_policy,
        "parent_bias_attention_enabled": bool(config.parent_bias_attention_enabled),
        "primary_task_name": event_meta_rows[0]["primary_task_name"] if event_meta_rows else None,
        "primary_metric_name": event_meta_rows[0]["primary_metric_name"] if event_meta_rows else None,
        "primary_class_names": primary_class_names,
        "missing_label_id": -1,
        "audit_sample_filename": f"audit_samples_{artifact_mode}.jsonl",
        "event_meta": event_meta_rows,
        "node_meta_sample": node_meta_sample,
        "truncation_drops": truncation_drops,
    }
    write_propagation_manifest(dataset_dir / f"propagation_manifest_{artifact_mode}.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    add_resolver_args(parser, include_mode=True, mode_required=False)
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--output_root", default="data/propagation_view")
    args = parser.parse_args()
    config = resolve_encoder_config("propagation", args.dataset_id, args.mode)
    build_propagation_arrays(args.dataset_id, args.mode, Path(args.input_root), Path(args.output_root), config=config)


if __name__ == "__main__":
    main()
