from __future__ import annotations

import torch
import torch.nn.functional as F

from models.fusion.schemas import FusionBatch, FusionOutput
from models.fusion.view_mask import ViewMask


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1))


def build_loss_weights(config) -> dict[str, float]:
    return {
        "view_reconstruction_loss_weight": float(config.view_reconstruction_loss_weight),
        "primary_task_loss_weight": float(config.primary_task_loss_weight),
        "stance_auxiliary_loss_weight": float(config.stance_auxiliary_loss_weight),
        "posterior_kl_loss_weight": float(config.posterior_kl_loss_weight),
    }


def masked_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0
    if not torch.any(valid):
        return logits.new_zeros(())
    return F.cross_entropy(logits[valid], labels[valid])


def masked_reconstruction_loss(
    prediction_mean: torch.Tensor,
    prediction_logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0
    if not torch.any(valid):
        return prediction_mean.new_zeros(())
    mean = prediction_mean[valid]
    logvar = prediction_logvar[valid]
    gold = target[valid]
    nll = 0.5 * (((gold - mean) ** 2) * torch.exp(-logvar) + logvar)
    return nll.mean()


def masked_aux_stance_loss(reply_logits: torch.Tensor | None, reply_labels: torch.Tensor, reply_mask: torch.Tensor) -> torch.Tensor:
    if reply_logits is None or reply_logits.numel() == 0:
        return reply_labels.new_zeros((), dtype=torch.float32)
    valid = reply_mask > 0
    if not torch.any(valid):
        return reply_logits.new_zeros(())
    return F.cross_entropy(reply_logits[valid], reply_labels[valid])


def compute_total_loss(
    output: FusionOutput, 
    batch: FusionBatch, 
    weights: dict[str, float],
    view_mask: ViewMask | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute total loss with optional view masking
    
    Args:
        output: FusionOutput from model forward pass
        batch: Input batch
        weights: Loss weight configuration
        view_mask: ViewMask to selectively include views in loss (None=all views)
    
    Returns:
        Dictionary of losses (per-view reconstructions, task loss, kl, total)
    """
    if view_mask is None:
        view_mask = ViewMask()  # All views active by default

    losses: dict[str, torch.Tensor] = {}
    raw_reconstruction_loss = output.z.new_zeros(())
    reconstruction_weight = weights.get("view_reconstruction_loss_weight", 1.0)
    task_primary_weight = weights.get("primary_task_loss_weight", 1.0)
    aux_stance_weight = weights.get("stance_auxiliary_loss_weight", 1.0)
    kl_weight = weights.get("posterior_kl_loss_weight", 1.0)
    loss_flags = output.loss_flags or {}
    enable_reconstruction = bool(loss_flags.get("reconstruction", True))
    enable_aux_stance = bool(loss_flags.get("aux_stance", True))
    enable_kl = bool(loss_flags.get("kl", True))
    
    for view_name, params in output.reconstruction_params.items():
        if enable_reconstruction:
            raw_view_loss = masked_reconstruction_loss(
                params["mean"],
                params["logvar"],
                params["target"],
                output.modality_posteriors[view_name].mask,
            )
        else:
            raw_view_loss = output.z.new_zeros(())
        if view_mask.is_active(view_name):
            raw_reconstruction_loss = raw_reconstruction_loss + raw_view_loss
        losses[f"reconstruction_{view_name}_loss"] = raw_view_loss * reconstruction_weight

    losses["reconstruction_loss"] = raw_reconstruction_loss * reconstruction_weight
    raw_task_primary_loss = masked_classification_loss(
        output.task_outputs["primary_task"]["logits"],
        batch.task_labels["primary_task"],
        batch.task_mask["primary_task"],
    )
    losses["task_primary_loss"] = raw_task_primary_loss * task_primary_weight
    if enable_aux_stance and view_mask.is_active("stance"):
        raw_aux_stance_loss = masked_aux_stance_loss(
            output.auxiliary_outputs.get("reply_logits"),
            batch.modalities["stance"]["reply_label_ids"],
            batch.modalities["stance"]["reply_label_mask"],
        )
    else:
        raw_aux_stance_loss = output.z.new_zeros(())
    losses["aux_stance_loss"] = raw_aux_stance_loss * aux_stance_weight
    if enable_kl:
        raw_kl_loss = kl_divergence_standard_normal(output.posterior_mu, output.posterior_logvar)
    else:
        raw_kl_loss = output.z.new_zeros(())
    losses["kl_loss"] = raw_kl_loss * kl_weight
    
    # Total loss only includes losses from active views
    losses["total_loss"] = (
        losses["reconstruction_loss"]
        + losses["task_primary_loss"]
        + losses["aux_stance_loss"]
        + losses["kl_loss"]
    )
    return losses
