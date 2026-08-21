from __future__ import annotations

import re
from typing import Any

from models.encoders.content_view.config import ContentViewConfig, Mode
from utils.text import approximate_token_len

URL_PATTERN = re.compile(r"(?i)\b((?:https?://|www\.)\S+)")
MENTION_PATTERN = re.compile(r"(?<![\w@])@([A-Za-z0-9_\-]+)")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([^\s#]+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)


def extract_source_text(event_row: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    text = (((event_row.get("content") or {}).get("text")) or {})
    quality = (text.get("quality") or {}).copy()
    candidate = text.get("norm_light") or text.get("raw") or ""
    return candidate, {
        "is_text_missing": bool(quality.get("is_text_missing", False)),
        "is_text_empty": bool(quality.get("is_text_empty", False)),
        "is_text_boilerplate": bool(quality.get("is_text_boilerplate", False)),
        "is_text_substantive": bool(quality.get("is_text_substantive", False)),
        "is_semantically_usable": bool(quality.get("is_semantically_usable", False)),
    }


def placeholderize_text(text: str, config: ContentViewConfig) -> str:
    value = text or ""
    if config.replace_urls:
        value = URL_PATTERN.sub("[URL]", value)
    if config.replace_mentions:
        value = MENTION_PATTERN.sub("[USER]", value)
    return value.strip()


def apply_prefix(text: str, config: ContentViewConfig, mode: Mode) -> str:
    _ = config
    _ = mode
    return text


def build_encoder_text(event_row: dict[str, Any], config: ContentViewConfig, mode: Mode) -> tuple[str, dict[str, bool]]:
    base_text, quality = extract_source_text(event_row)
    placeholderized = placeholderize_text(base_text, config)
    return apply_prefix(placeholderized, config, mode), quality


def count_mentions(text: str) -> int:
    return text.count("[USER]")


def count_hashtags(text: str) -> int:
    return len(HASHTAG_PATTERN.findall(text))


def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(text))


def count_exclamations(text: str) -> int:
    return text.count("!")


def count_questions(text: str) -> int:
    return text.count("?")


def encoder_text_lengths(text: str) -> tuple[int, int]:
    return len(text), approximate_token_len(text)
