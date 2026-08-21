from __future__ import annotations

from typing import Any

from models.encoders.propagation_view.config import PropagationViewConfig

NODE_SAMPLING_ROLE_ALL_KEPT = 0
NODE_SAMPLING_ROLE_SOURCE = 1
NODE_SAMPLING_ROLE_EARLY = 2
NODE_SAMPLING_ROLE_MIDDLE_POOL = 3
NODE_SAMPLING_ROLE_LATE = 4


def _equally_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    raw = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    result: list[int] = []
    for idx in raw:
        candidate = max(0, min(length - 1, int(idx)))
        while candidate in seen and candidate + 1 < length:
            candidate += 1
        while candidate in seen and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    cursor = 0
    while len(result) < count and cursor < length:
        if cursor not in seen:
            seen.add(cursor)
            result.append(cursor)
        cursor += 1
    return sorted(result)


def propagation_mode_for_nodes(nodes: list[dict[str, Any]]) -> str:
    non_source = [node for node in nodes if not node.get("is_source")]
    if not non_source:
        return "missing"
    observed = 0
    for node in non_source:
        node_time = node.get("time") or {}
        kind = node_time.get("observation_kind")
        if kind in {"absolute_timestamp", "relative_delay_observed"}:
            observed += 1
    ratio = observed / max(1, len(non_source))
    max_depth = max((node.get("depth") or 0) for node in non_source)
    if ratio >= 0.7 and max_depth >= 2:
        return "strong"
    if ratio >= 0.3:
        return "medium"
    return "sparse"


def _sort_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_nodes = [node for node in nodes if node.get("is_source")]
    source_nodes.sort(key=lambda node: node["post_id"])
    non_source_nodes = [node for node in nodes if not node.get("is_source")]
    non_source_nodes.sort(key=lambda node: node["_time_sort_key"])
    return source_nodes[:1] + non_source_nodes


def _contiguous_center_window(nodes: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    if window_size <= 0 or not nodes:
        return []
    count = min(int(window_size), len(nodes))
    start = max(0, (len(nodes) - count) // 2)
    return nodes[start : start + count]


def select_and_order_real_nodes(
    nodes: list[dict[str, Any]],
    config: PropagationViewConfig,
    *,
    max_nodes_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not nodes:
        return [], {"dropped_by_truncation": 0}
    ordered = _sort_nodes(nodes)
    max_nodes = int(max_nodes_override if max_nodes_override is not None else config.max_nodes_per_event_value())
    if len(ordered) <= max_nodes:
        for idx, node in enumerate(ordered):
            node["_keep_reason"] = "source" if idx == 0 and node.get("is_source") else "all_kept"
            node["_sampling_role_id"] = NODE_SAMPLING_ROLE_SOURCE if idx == 0 and node.get("is_source") else NODE_SAMPLING_ROLE_ALL_KEPT
        return ordered, {"dropped_by_truncation": 0}

    source = ordered[:1]
    remainder = ordered[1:]
    remaining_budget = max(0, max_nodes - len(source))
    early_budget = int(remaining_budget * config.early_fraction)
    middle_budget = int(remaining_budget * config.middle_fraction)
    late_budget = max(0, remaining_budget - early_budget - middle_budget)
    middle_pool_budget = max(
        middle_budget,
        int(round(float(middle_budget) * float(config.middle_sampling_pool_multiplier))),
    )

    selected: list[dict[str, Any]] = []
    for node in source:
        node["_keep_reason"] = "source"
        node["_sampling_role_id"] = NODE_SAMPLING_ROLE_SOURCE
    selected.extend(source)

    early_nodes = remainder[:early_budget]
    late_nodes = remainder[-late_budget:] if late_budget > 0 else []
    for node in early_nodes:
        node["_keep_reason"] = "early_window"
        node["_sampling_role_id"] = NODE_SAMPLING_ROLE_EARLY
    selected.extend(early_nodes)

    reserved_ids = {node["post_id"] for node in selected}
    reserved_ids.update(node["post_id"] for node in late_nodes)
    middle_candidates = [node for node in remainder if node["post_id"] not in reserved_ids]
    middle_nodes = _contiguous_center_window(middle_candidates, min(middle_pool_budget, len(middle_candidates)))
    for node in middle_nodes:
        node["_keep_reason"] = "middle_candidate_pool"
        node["_sampling_role_id"] = NODE_SAMPLING_ROLE_MIDDLE_POOL
    selected.extend(middle_nodes)
    for node in late_nodes:
        node["_keep_reason"] = "late_window"
        node["_sampling_role_id"] = NODE_SAMPLING_ROLE_LATE
    selected.extend(late_nodes)

    dedup: dict[str, dict[str, Any]] = {}
    for node in selected:
        dedup[node["post_id"]] = node
    final_nodes = sorted(dedup.values(), key=lambda node: node["_time_sort_key"])
    if final_nodes and not final_nodes[0].get("is_source"):
        source_nodes = [node for node in ordered if node.get("is_source")]
        if source_nodes:
            final_nodes = [source_nodes[0]] + [node for node in final_nodes if node["post_id"] != source_nodes[0]["post_id"]]
    return final_nodes, {"dropped_by_truncation": max(0, len(nodes) - len(final_nodes))}


def select_and_order_nodes(nodes: list[dict[str, Any]], config: PropagationViewConfig, mode: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _ = mode
    return select_and_order_real_nodes(nodes, config)
