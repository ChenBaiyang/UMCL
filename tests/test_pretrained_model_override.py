from __future__ import annotations

from config.resolver import (
    reset_pretrained_model_override,
    resolve_encoder_config,
    set_pretrained_model_override,
)


def test_pretrained_model_override_applies_to_text_encoders_only() -> None:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    token = set_pretrained_model_override(model_name)
    try:
        content = resolve_encoder_config("content", "PHEME", "full")
        stance = resolve_encoder_config("stance", "PHEME", "full")
        propagation = resolve_encoder_config("propagation", "PHEME", "full")
    finally:
        reset_pretrained_model_override(token)

    assert content.model_name("full") == model_name
    assert stance.model_name("full") == model_name
    assert "model_name_value" not in propagation.resolved_overrides


def test_pretrained_model_override_is_reset() -> None:
    configured_model = resolve_encoder_config("content", "PHEME", "full").model_name("full")
    token = set_pretrained_model_override("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reset_pretrained_model_override(token)

    assert resolve_encoder_config("content", "PHEME", "full").model_name("full") == configured_model
