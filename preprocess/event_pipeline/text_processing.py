from __future__ import annotations

from preprocess.event_pipeline.schema import TextBundle
from utils.text import approximate_token_len, light_normalize_text


def preprocess_text_bundle(text: TextBundle) -> TextBundle:
    if text.raw is None:
        text.norm_light = None
        text.char_len = None
        text.token_len_approx = None
        text.quality.is_text_missing = True
        text.quality.is_text_placeholder = text.status == "placeholder"
        text.quality.is_text_boilerplate = text.status == "boilerplate"
        text.quality.is_semantically_usable = False
        return text

    text.norm_light = light_normalize_text(text.raw)
    text.char_len = len(text.norm_light)
    text.token_len_approx = approximate_token_len(text.norm_light)
    text.quality.is_text_missing = False
    text.quality.is_text_empty = text.norm_light == ""
    text.quality.is_text_placeholder = text.status == "placeholder"
    text.quality.is_text_boilerplate = text.status == "boilerplate"
    text.quality.is_text_substantive = text.status == "substantive"
    text.quality.is_semantically_usable = text.status == "substantive" or (
        text.status == "unknown" and bool(text.norm_light)
    )
    return text
