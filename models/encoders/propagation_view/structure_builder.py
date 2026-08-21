from __future__ import annotations

from collections import Counter


def depth_bucket_id(depth: int | None) -> int:
    if depth is None or depth < 0:
        return 0
    if depth == 1:
        return 1
    if depth == 2:
        return 2
    if depth >= 3:
        return 3
    return 0


def order_bucket_id(node_order_index: int | None) -> int:
    if node_order_index is None or node_order_index < 0:
        return 0
    if node_order_index <= 2:
        return 1
    if node_order_index <= 8:
        return 2
    return 3


def compute_child_counts(nodes: list[dict]) -> dict[str, int]:
    counter = Counter()
    for node in nodes:
        parent_post_id = node.get("parent_post_id")
        if parent_post_id:
            counter[parent_post_id] += 1
    return dict(counter)
