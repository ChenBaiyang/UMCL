from __future__ import annotations

import math
import re
from typing import Any

from models.encoders.stance_view.relation_builder import normalize_social_text

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HASHTAG_RE = re.compile(r"#\S+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")


def depth_bucket_id(depth: int | None) -> int:
    if depth is None or depth < 0:
        return 0
    if depth <= 1:
        return 1
    if depth <= 3:
        return 2
    return 3


def time_bucket_id(reply_time: dict[str, Any]) -> int:
    minutes = reply_time.get("time_from_source_min")
    if minutes is None:
        values = reply_time.get("time_delay_minutes_raw_values") or []
        if values:
            try:
                minutes = float(values[0])
            except (TypeError, ValueError):
                minutes = None
    if minutes is None:
        return 0
    if minutes <= 30:
        return 1
    if minutes <= 360:
        return 2
    return 3


def interaction_type_id(interaction_type: str | None) -> int:
    return {"unknown": 0, "reply": 1, "repost": 2, "quote_repost": 3, "source": 4}.get(interaction_type or "unknown", 0)


def build_context_cat(reply: dict[str, Any]) -> list[int]:
    return [
        depth_bucket_id(reply.get("depth")),
        time_bucket_id(reply.get("time") or {}),
        1 if reply.get("parent_is_source") else 0,
        interaction_type_id(reply.get("interaction_type")),
    ]


def _tokenize_for_overlap(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(normalize_social_text(text)) if token.strip()]


def _overlap_ratio(a: str, b: str) -> float:
    tokens_a = _tokenize_for_overlap(a)
    tokens_b = set(_tokenize_for_overlap(b))
    if not tokens_a:
        return 0.0
    overlap = sum(1 for token in tokens_a if token in tokens_b)
    return float(overlap) / float(len(tokens_a))


def _text_bundle_value(bundle: dict[str, Any]) -> str:
    return str(bundle.get("norm_light") or bundle.get("raw") or "")


def build_context_dense(
    reply: dict[str, Any],
    source_text_bundle: dict[str, Any],
    parent_text_bundle: dict[str, Any] | None,
    semantic_similarity: dict[str, float] | None = None,
) -> list[float]:
    reply_text_bundle = reply.get("text") or {}
    reply_text = _text_bundle_value(reply_text_bundle)
    source_text = _text_bundle_value(source_text_bundle)
    parent_text = _text_bundle_value(parent_text_bundle or {})
    reply_time = reply.get("time") or {}
    minutes = reply_time.get("time_from_source_min")
    if minutes is None:
        values = reply_time.get("time_delay_minutes_raw_values") or []
        if values:
            try:
                minutes = float(values[0])
            except (TypeError, ValueError):
                minutes = 0.0
    minutes = float(minutes or 0.0)
    text_quality = reply_text_bundle.get("quality") or {}
    char_len = reply_text_bundle.get("char_len")
    if char_len is None:
        char_len = len(reply_text)
    token_len_approx = reply_text_bundle.get("token_len_approx")
    if token_len_approx is None:
        token_len_approx = len(_tokenize_for_overlap(reply_text))
    normalized_reply = normalize_social_text(reply_text)
    emoji_count = min(3, len(EMOJI_RE.findall(reply_text)))
    question_count = min(3, reply_text.count("?"))
    exclamation_count = min(3, reply_text.count("!"))
    has_url = 1.0 if "[URL]" in normalized_reply or "http" in reply_text.lower() else 0.0
    has_mention = 1.0 if "[USER]" in normalized_reply or "@" in reply_text else 0.0
    has_hashtag = 1.0 if HASHTAG_RE.search(reply_text) else 0.0
    is_short_reply = 1.0 if int(char_len or 0) <= 30 else 0.0
    is_very_short_reply = 1.0 if int(char_len or 0) <= 10 else 0.0
    has_parent_text = 1.0 if parent_text_bundle and (text_quality.get("is_semantically_usable", True) or parent_text) else 0.0
    semantic_similarity = semantic_similarity or {}
    return [
        float(math.log1p(max(0.0, minutes))),
        float(math.log1p(max(0, int(char_len or 0)))),
        float(math.log1p(max(0, int(token_len_approx or 0)))),
        float(semantic_similarity.get("reply_source_cosine", 0.0)),
        float(semantic_similarity.get("reply_parent_cosine", 0.0)) if has_parent_text else 0.0,
        float(semantic_similarity.get("parent_source_cosine", 0.0)) if has_parent_text else 0.0,
        float(_overlap_ratio(reply_text, source_text)),
        float(_overlap_ratio(reply_text, parent_text)) if has_parent_text else 0.0,
        has_parent_text,
        has_url,
        has_mention,
        has_hashtag,
        float(emoji_count),
        float(question_count),
        float(exclamation_count),
        is_short_reply,
        is_very_short_reply,
    ]
