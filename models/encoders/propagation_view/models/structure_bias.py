from __future__ import annotations

import torch


def build_parent_attention_bias(
    parent_index: torch.Tensor,
    valid_mask: torch.Tensor,
    num_heads: int,
    parent_bias_value: float,
    source_bias_value: float,
) -> torch.Tensor:
    batch_size, seq_len = parent_index.shape
    if batch_size == 0 or seq_len == 0:
        return torch.zeros((batch_size * num_heads, seq_len, seq_len), dtype=torch.float32, device=parent_index.device)

    bias = torch.full((batch_size, seq_len, seq_len), -1.0e4, dtype=torch.float32, device=parent_index.device)
    valid_pair = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(1)
    bias.masked_fill_(valid_pair, 0.0)

    source_valid = valid_mask & valid_mask[:, :1]
    bias[:, :, 0].add_(source_bias_value * source_valid.to(bias.dtype))

    parent_in_bounds = (parent_index >= 0) & (parent_index < seq_len)
    if parent_in_bounds.any():
        clamped_parent = parent_index.clamp(min=0, max=seq_len - 1)
        parent_valid = parent_in_bounds & valid_mask.gather(1, clamped_parent)
        if parent_valid.any():
            batch_idx, node_idx = torch.nonzero(parent_valid, as_tuple=True)
            bias[batch_idx, node_idx, clamped_parent[batch_idx, node_idx]] += parent_bias_value

    return bias.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(batch_size * num_heads, seq_len, seq_len).contiguous()
