from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from transformers import AutoModel

from models.encoders.content_view.config import ContentViewConfig, Mode


def _base_model(model: nn.Module) -> nn.Module:
    return getattr(model, "base_model", model)


def _encoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    base = _base_model(model)
    encoder = getattr(base, "encoder", None)
    if encoder is None:
        return []
    layers = getattr(encoder, "layer", None)
    if layers is not None:
        return list(layers)
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        return list(layers)
    return []


def _freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def _freeze_except_top_k_layers(model: nn.Module, top_k_layers: int) -> None:
    layers = _encoder_layers(model)
    keep = max(0, min(len(layers), int(top_k_layers)))
    cutoff = max(0, len(layers) - keep)
    for layer in layers[:cutoff]:
        _freeze_module(layer)


def load_backbone(config: ContentViewConfig, mode: Mode) -> tuple[nn.Module, int]:
    model = AutoModel.from_pretrained(config.model_name(mode))
    hidden_size = int(model.config.hidden_size)
    base = _base_model(model)
    if config.freeze_embeddings:
        _freeze_module(getattr(base, "embeddings", None))
    if mode == "lite":
        _freeze_except_top_k_layers(model, config.train_top_k_layers)
    elif mode == "full":
        if config.tuning_strategy == "train_top_k_layers":
            _freeze_except_top_k_layers(model, config.train_top_k_layers)
        elif config.tuning_strategy != "full_finetune":
            raise ValueError(f"Unsupported full tuning strategy: {config.tuning_strategy}")
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return model, hidden_size
