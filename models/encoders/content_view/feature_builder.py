from __future__ import annotations

from typing import Any

import numpy as np

from models.encoders.content_view.config import ContentViewConfig
from models.encoders.content_view.text_builder import (
    count_emojis,
    count_exclamations,
    count_hashtags,
    count_mentions,
    count_questions,
)


def build_surface_feature_vector(
    event_row: dict[str, Any],
    encoder_text: str,
    config: ContentViewConfig,
    token_len_approx: int,
) -> np.ndarray:
    content = event_row.get("content") or {}
    text = content.get("text") or {}
    surface = content.get("surface") or {}

    char_len = int(text.get("char_len") or len(encoder_text))
    features = {
        "has_url": 1.0 if bool(surface.get("has_url")) else 0.0,
        "has_media": 1.0 if bool(surface.get("has_media")) else 0.0,
        "char_len": float(char_len),
        "token_len_approx": float(token_len_approx),
        "mention_count": float(count_mentions(encoder_text)),
        "hashtag_count": float(count_hashtags(encoder_text)),
        "emoji_count": float(count_emojis(encoder_text)),
        "exclamation_count": float(count_exclamations(encoder_text)),
        "question_count": float(count_questions(encoder_text)),
    }
    return np.asarray([features[name] for name in config.surface_feature_names], dtype=np.float32)
