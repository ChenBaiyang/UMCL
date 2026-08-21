from __future__ import annotations

import math

import torch
import torch.nn as nn


class ParentBiasedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        parent_index: torch.Tensor | None = None,
        *,
        parent_bias_value: float = 0.0,
        source_bias_value: float = 0.0,
        parent_bias_attention_enabled: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _hidden_dim = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        key_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        logits = logits.masked_fill(~key_mask, float("-inf"))
        query_mask = valid_mask.unsqueeze(1).unsqueeze(-1)
        logits = logits.masked_fill(~query_mask, float("-inf"))

        if parent_bias_attention_enabled and parent_index is not None and seq_len > 0:
            parent_bias = torch.zeros((batch_size, seq_len, seq_len), dtype=logits.dtype, device=logits.device)
            valid_parent = (parent_index >= 0) & valid_mask
            if valid_parent.any():
                batch_ids, child_ids = torch.nonzero(valid_parent, as_tuple=True)
                parent_ids = parent_index[batch_ids, child_ids]
                parent_bias[batch_ids, child_ids, parent_ids] = float(parent_bias_value)
            source_bias = torch.zeros_like(parent_bias)
            source_bias[:, :, 0] = float(source_bias_value)
            source_bias = source_bias.masked_fill(~valid_mask.unsqueeze(1), 0.0)
            logits = logits + parent_bias.unsqueeze(1) + source_bias.unsqueeze(1)

        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)
        hidden = torch.matmul(attn, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(hidden)


class SequenceTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = ParentBiasedSelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        parent_index: torch.Tensor | None = None,
        *,
        parent_bias_value: float = 0.0,
        source_bias_value: float = 0.0,
        parent_bias_attention_enabled: bool = False,
    ) -> torch.Tensor:
        attn_out = self.attn(
            x,
            valid_mask,
            parent_index,
            parent_bias_value=parent_bias_value,
            source_bias_value=source_bias_value,
            parent_bias_attention_enabled=parent_bias_attention_enabled,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SequenceTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, ffn_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SequenceTransformerBlock(hidden_dim, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        parent_index: torch.Tensor | None = None,
        *,
        parent_bias_value: float = 0.0,
        source_bias_value: float = 0.0,
        parent_bias_attention_enabled: bool = False,
    ) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(
                hidden,
                valid_mask,
                parent_index,
                parent_bias_value=parent_bias_value,
                source_bias_value=source_bias_value,
                parent_bias_attention_enabled=parent_bias_attention_enabled,
            )
        return hidden
