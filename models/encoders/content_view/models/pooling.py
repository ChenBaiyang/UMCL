from __future__ import annotations

import torch


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def pretrained_s_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def _pretrained_s_available(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> bool:
    return (
        last_hidden_state.ndim == 3
        and attention_mask.ndim == 2
        and last_hidden_state.shape[0] == attention_mask.shape[0]
        and last_hidden_state.shape[1] == attention_mask.shape[1]
        and last_hidden_state.shape[1] > 0
    )


def pool_sentence(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "masked_mean_pooling":
        return masked_mean_pool(last_hidden_state, attention_mask)
    if mode == "pretrained_s_pooling":
        mean_pooled = masked_mean_pool(last_hidden_state, attention_mask)
        if not _pretrained_s_available(last_hidden_state, attention_mask):
            return mean_pooled
        s_pooled = pretrained_s_pool(last_hidden_state)
        s_available = attention_mask[:, 0].to(dtype=torch.bool).unsqueeze(-1)
        return torch.where(s_available, s_pooled, mean_pooled)
    raise ValueError(
        "Unsupported content_sentence_pool_mode "
        f"{mode!r}; expected one of: masked_mean_pooling, pretrained_s_pooling"
    )
