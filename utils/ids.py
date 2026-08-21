from __future__ import annotations


def stable_sort_key(*parts: object) -> tuple[str, ...]:
    return tuple("" if part is None else str(part) for part in parts)
