from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["lite", "full"]


@dataclass
class ContentViewConfig:
    dataset_id: str | None = None
    model_name_value: str = "xlm-roberta-base"
    text_source_policy: str = "placeholderized_norm_light"
    replace_urls: bool = True
    replace_mentions: bool = True
    preserve_hashtags: bool = True
    preserve_emojis: bool = True
    build_surface_features: bool = True
    surface_feature_names: list[str] = field(
        default_factory=lambda: [
            "has_url",
            "has_media",
            "char_len",
            "token_len_approx",
            "mention_count",
            "hashtag_count",
            "emoji_count",
            "exclamation_count",
            "question_count",
        ]
    )
    storage_backend: str = "npz_compressed"
    shard_size: int = 2048
    content_sentence_pool_mode: Literal["masked_mean_pooling", "pretrained_s_pooling"] = "pretrained_s_pooling"
    enable_surface_fusion: bool = True
    max_length_value: int = 64
    event_hidden_dim: int = 256
    surface_feature_hidden_dim: int = 64
    dropout: float = 0.1
    freeze_embeddings: bool = True
    train_top_k_layers: int = 4
    tuning_strategy: str = "full_finetune"
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def model_name(self, mode: Mode) -> str:
        _ = mode
        return str(self.model_name_value)

    def max_length(self, mode: Mode) -> int:
        _ = mode
        return int(self.max_length_value)

    def artifact_mode(self, mode: Mode) -> Literal["lite", "full"]:
        return "full" if mode == "full" else "lite"

    def as_dict(self) -> dict:
        return asdict(self)
