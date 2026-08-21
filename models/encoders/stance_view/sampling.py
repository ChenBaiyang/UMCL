from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SamplingResult:
    selected_replies: list[dict[str, Any]]
    original_usable_reply_count: int
    sampled_reply_count: int
    truncated: bool
    sampling_positions: dict[str, str]


def _equally_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    raw = [round(idx * (length - 1) / (count - 1)) for idx in range(count)]
    seen: set[int] = set()
    selected: list[int] = []
    for idx in raw:
        candidate = max(0, min(length - 1, int(idx)))
        while candidate in seen and candidate + 1 < length:
            candidate += 1
        while candidate in seen and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    cursor = 0
    while len(selected) < count and cursor < length:
        if cursor not in seen:
            seen.add(cursor)
            selected.append(cursor)
        cursor += 1
    return sorted(selected)


def _allocate_region_budget(total_budget: int, region_lengths: list[int]) -> list[int]:
    if total_budget <= 0:
        return [0 for _ in region_lengths]
    base = total_budget // max(1, len(region_lengths))
    allocation = [min(length, base) for length in region_lengths]
    assigned = sum(allocation)
    remaining = total_budget - assigned
    order = sorted(range(len(region_lengths)), key=lambda idx: (-region_lengths[idx], idx))
    while remaining > 0:
        progressed = False
        for idx in order:
            if allocation[idx] < region_lengths[idx]:
                allocation[idx] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return allocation


def sample_replies_deterministically(replies: list[dict[str, Any]], max_replies_per_event: int) -> SamplingResult:
    ordered = list(replies)
    total = len(ordered)
    if total <= max_replies_per_event:
        positions = {str(reply["post_id"]): "unsampled_all" for reply in ordered}
        return SamplingResult(ordered, total, total, False, positions)

    early_end = total // 3
    late_start = total - (total // 3)
    middle_end = late_start
    regions: list[tuple[str, list[dict[str, Any]]]] = [
        ("early", ordered[:early_end]),
        ("middle", ordered[early_end:middle_end]),
        ("late", ordered[late_start:]),
    ]
    region_lengths = [len(items) for _, items in regions]
    region_budget = _allocate_region_budget(int(max_replies_per_event), region_lengths)

    selected_pairs: list[tuple[int, str]] = []
    offset = 0
    for (region_name, items), budget in zip(regions, region_budget):
        if budget > 0:
            for local_idx in _equally_spaced_indices(len(items), budget):
                selected_pairs.append((offset + local_idx, region_name))
        offset += len(items)

    selected_pairs.sort(key=lambda item: item[0])
    selected_replies = [ordered[idx] for idx, _ in selected_pairs]
    positions = {str(ordered[idx]["post_id"]): region_name for idx, region_name in selected_pairs}
    return SamplingResult(selected_replies, total, len(selected_replies), True, positions)
