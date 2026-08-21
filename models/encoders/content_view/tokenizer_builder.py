from __future__ import annotations

from transformers import AutoTokenizer

from models.encoders.content_view.config import ContentViewConfig, Mode


def build_tokenizer(config: ContentViewConfig, mode: Mode):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name(mode))
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def tokenize_texts(tokenizer, texts: list[str], config: ContentViewConfig, mode: Mode) -> tuple[list[list[int]], list[list[int]]]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length(mode),
        return_attention_mask=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]
