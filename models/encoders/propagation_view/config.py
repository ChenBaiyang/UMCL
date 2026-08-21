from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["lite", "full"]
PropagationFamily = Literal["propagation_view"]


@dataclass
class PropagationViewConfig:
    dataset_id: str | None = None
    max_nodes_per_event: int = 512
    categorical_feature_embedding_dim: int = 24
    continuous_feature_hidden_dim: int = 64
    structure_feature_hidden_dim: int = 64
    node_hidden_dim: int = 512
    transformer_layer_count: int = 4
    attention_head_count: int = 8
    transformer_ffn_hidden_dim: int = 2048
    dropout: float = 0.15
    event_hidden_dim: int = 128
    aux_node_head_enabled: bool = False
    shard_size: int = 32768
    enable_order_bucket: bool = False
    early_fraction: float = 1.0 / 3.0
    middle_fraction: float = 1.0 / 3.0
    late_fraction: float = 1.0 / 3.0
    middle_sampling_pool_multiplier: float = 2.0
    event_shape_features_enabled: bool = True
    temporal_profile_enabled: bool = True
    temporal_profile_policy: str = "multi_resolution"
    event_summary_fusion_enabled: bool = True
    pooling_policy: str = "root_mean_max_stage"
    stage_pooling_enabled: bool = True
    clean_redundant_continuous_features: bool = True
    local_structure_features_policy: str = "balanced"
    parent_bias_attention_enabled: bool = True
    parent_bias_relation_policy: str = "parent_source"
    parent_bias_value: float = 1.5
    source_bias_value: float = 0.75
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def max_nodes_per_event_value(self) -> int:
        return int(self.max_nodes_per_event)

    def max_nodes(self, mode: Mode) -> int:
        _ = mode
        return self.max_nodes_per_event_value()

    def hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.node_hidden_dim)

    def transformer_layers(self, mode: Mode) -> int:
        _ = mode
        return int(self.transformer_layer_count)

    def attention_heads(self, mode: Mode) -> int:
        _ = mode
        return int(self.attention_head_count)

    def dropout_rate(self, mode: Mode) -> float:
        _ = mode
        return float(self.dropout)

    def transformer_ffn_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.transformer_ffn_hidden_dim)

    def artifact_mode(self, mode: Mode) -> Literal["lite", "full"]:
        return "full" if mode == "full" else "lite"

    def as_dict(self) -> dict:
        return asdict(self)
