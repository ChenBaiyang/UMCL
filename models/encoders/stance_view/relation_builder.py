from __future__ import annotations

import re
from typing import Any

from models.encoders.stance_view.config import Mode, StanceViewConfig

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?<!\w)@[\w_]+")


def _text_value(text_bundle: dict[str, Any]) -> str:
    return str(text_bundle.get("norm_light") or text_bundle.get("raw") or "").strip()


def normalize_social_text(text: str) -> str:
    if not text:
        return ""
    text = URL_RE.sub("[URL]", text)
    text = MENTION_RE.sub("[USER]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_source_aware_text(
    source_text_bundle: dict[str, Any],
    reply_text_bundle: dict[str, Any],
    config: StanceViewConfig,
    mode: Mode,
) -> str:
    source_text = normalize_social_text(_text_value(source_text_bundle))
    reply_text = normalize_social_text(_text_value(reply_text_bundle))
    relation = f"[SRC] {source_text} [RPL] {reply_text}".strip()
    if config.artifact_mode(mode) == "lite":
        return f"passage: {relation}"
    return relation
