"""
Training wrapper that supports view masking for selective modality testing
"""
from typing import Any, Callable
import torch
from models.fusion.losses import compute_total_loss as compute_total_loss_impl
from models.fusion.view_mask import ViewMask


def create_masked_forward_fn(model: torch.nn.Module, view_mask: ViewMask | None) -> Callable:
    """Create a forward function that applies view masking"""
    def forward(batch):
        return model(batch, view_mask=view_mask)
    return forward


def create_masked_loss_fn(
    view_mask: ViewMask | None,
) -> Callable:
    """Create a loss function that applies view masking"""
    def compute_loss(output, batch, weights):
        return compute_total_loss_impl(output, batch, weights, view_mask=view_mask)
    return compute_loss


def print_view_mask_info(view_mask: ViewMask | None) -> str:
    """Generate informative string about view mask configuration"""
    if view_mask is None:
        view_mask = ViewMask()
    
    active = ", ".join(view_mask.get_active_views())
    inactive = ", ".join(view_mask.get_inactive_views())
    coverage = 100 * len(view_mask.active_views) / len(view_mask.all_views)
    
    lines = [
        "\n" + "="*80,
        "MODALITY VIEW MASKING CONFIGURATION",
        "="*80,
        f"Active views ({len(view_mask.active_views)}/{len(view_mask.all_views)}): {active}",
        f"Inactive views ({len(view_mask.get_inactive_views())}): {inactive}",
        f"Coverage: {coverage:.0f}%",
        "="*80 + "\n",
    ]
    
    return "\n".join(lines)
