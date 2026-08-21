from __future__ import annotations

from typing import Any


def canonical_text_status(reply: dict[str, Any]) -> str:
    text = reply.get("text") or {}
    quality = text.get("quality") or {}
    if reply.get("is_placeholder"):
        return "placeholder"
    if not reply.get("is_observed_text"):
        return "unobserved"
    if quality.get("is_text_empty"):
        return "empty"
    if reply.get("is_boilerplate_text"):
        return "boilerplate"
    if not reply.get("is_semantically_substantive"):
        return "non_substantive"
    return "substantive"


def normalize_interaction_type(value: str | None) -> str:
    if value in {"reply", "repost", "quote_repost", "source"}:
        return value
    return "unknown"


def classify_reply_for_stance(dataset_id: str, reply: dict[str, Any]) -> tuple[bool, str]:
    status = canonical_text_status(reply)
    interaction_type = normalize_interaction_type(reply.get("interaction_type"))
    if status != "substantive":
        return False, status
    if dataset_id in {"Twitter15", "Twitter16"}:
        return False, "dataset_zero_reply"
    if dataset_id == "Weibo":
        if interaction_type == "repost":
            return False, "repost_only"
        return True, "kept_weibo_substantive"
    return True, "kept_substantive"


def keep_reply_for_stance(dataset_id: str, reply: dict[str, Any]) -> bool:
    keep, _ = classify_reply_for_stance(dataset_id, reply)
    return keep
