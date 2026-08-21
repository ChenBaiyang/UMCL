from __future__ import annotations


def interaction_type_id(value: str | None) -> int:
    return {"unknown": 0, "source": 1, "reply": 2, "repost": 3, "quote_repost": 4}.get(value or "unknown", 0)


def text_status_id(value: str | None) -> int:
    return {"unknown": 0, "substantive": 1, "boilerplate": 2, "empty": 3, "placeholder": 4}.get(value or "unknown", 0)


def is_boilerplate_text(value: str | None) -> int:
    return 1 if value == "boilerplate" else 0
