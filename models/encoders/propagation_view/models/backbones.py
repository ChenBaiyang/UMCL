from __future__ import annotations

from models.encoders.propagation_view.config import Mode, PropagationViewConfig


def propagation_model_dims(config: PropagationViewConfig, mode: Mode) -> dict[str, int | float]:
    return {
        "hidden_dim": config.hidden_dim(mode),
        "dropout": config.dropout_rate(mode),
        "layers": config.transformer_layers(mode),
        "heads": config.attention_heads(mode),
        "ffn_hidden_dim": config.transformer_ffn_hidden_dim_value(mode),
        "max_nodes_per_event": config.max_nodes_per_event_value(),
    }
