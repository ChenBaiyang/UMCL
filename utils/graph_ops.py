"""Native PyTorch pooling shared by graph baselines."""

import torch


def graph_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean of floating node features [N, F] per graph; empty groups are zero.

    Zero-valued (including masked) nodes still count in the denominator.
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    total = x.new_zeros((num_graphs, x.size(1))).index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=num_graphs).to(dtype=x.dtype)
    return total / count.clamp_min(1).unsqueeze(1)
