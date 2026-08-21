from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["lite", "full"]
ArtifactMode = Literal["lite", "full"]
StanceFamily = Literal["stance_view"]


@dataclass
class StanceViewConfig:
    dataset_id: str | None = None
    model_name_value: str = "xlm-roberta-base"
    max_length_value: int = 64
    encode_batch_size: int = 64
    shard_size: int = 4096
    max_replies_per_event: int = 256
    reply_embedding_dim: int = 768
    reply_embedding_proj_dim_value: int = 192
    context_hidden_dim_value: int = 48
    dense_hidden_dim_value: int = 48
    reply_mlp_hidden_dim_value: int = 224
    reply_hidden_dim_value: int = 192
    summarizer_hidden_dim_value: int = 384
    event_hidden_dim: int = 256
    context_cat_fields: list[str] = field(
        default_factory=lambda: ["depth_bucket_id", "time_bucket_id", "parent_is_source", "interaction_type_id"]
    )
    context_dense_fields: list[str] = field(
        default_factory=lambda: [
            "log1p_time_from_source_min",
            "char_len_log1p",
            "token_len_approx_log1p",
            "reply_source_cosine",
            "reply_parent_cosine",
            "parent_source_cosine",
            "source_overlap_ratio",
            "parent_overlap_ratio",
            "has_parent_text",
            "has_url",
            "has_mention",
            "has_hashtag",
            "emoji_count_clipped",
            "question_count_clipped",
            "exclamation_count_clipped",
            "is_short_reply",
            "is_very_short_reply",
        ]
    )
    stance_label_names: list[str] = field(default_factory=lambda: ["support", "deny", "query", "comment"])
    enable_reply_classifier_head: bool = False
    dropout: float = 0.1
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def artifact_mode(self, mode: Mode) -> ArtifactMode:
        return "full" if mode == "full" else "lite"

    def model_name(self, mode: Mode) -> str:
        _ = mode
        return str(self.model_name_value)

    def max_length(self, mode: Mode) -> int:
        _ = mode
        return int(self.max_length_value)

    def reply_embedding_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.reply_embedding_dim)

    def reply_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.reply_hidden_dim_value)

    def reply_embedding_proj_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.reply_embedding_proj_dim_value)

    def context_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.context_hidden_dim_value)

    def dense_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.dense_hidden_dim_value)

    def reply_mlp_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.reply_mlp_hidden_dim_value)

    def summarizer_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.summarizer_hidden_dim_value)

    def encode_batch_size_value(self, override: int | None = None) -> int:
        if override is not None:
            return max(1, int(override))
        return int(self.encode_batch_size)

    def max_replies_per_event_value(self) -> int:
        return int(self.max_replies_per_event)

    def as_dict(self) -> dict:
        return asdict(self)
