from __future__ import annotations

import torch


def clamp_logvar(logvar: torch.Tensor, min_value: float = -8.0, max_value: float = 8.0) -> torch.Tensor:
    return torch.clamp(logvar, min=min_value, max=max_value)


def product_of_experts(
    mus: list[torch.Tensor],
    logvars: list[torch.Tensor],
    masks: list[torch.Tensor],
    expert_weights: list[torch.Tensor] | None = None,
    use_unit_prior_expert: bool = True,
    logvar_min: float = -8.0,
    logvar_max: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not mus:
        raise ValueError("product_of_experts requires at least one modality posterior.")
    if not (len(mus) == len(logvars) == len(masks)):
        raise ValueError("mus, logvars, and masks must have the same length.")
    if expert_weights is not None and len(expert_weights) != len(mus):
        raise ValueError("expert_weights must match the number of experts.")
    reference = mus[0]
    batch_size, posterior_latent_dim = reference.shape
    device = reference.device
    dtype = reference.dtype

    total_precision = torch.zeros((batch_size, posterior_latent_dim), dtype=dtype, device=device)
    weighted_mean = torch.zeros((batch_size, posterior_latent_dim), dtype=dtype, device=device)
    if use_unit_prior_expert: ## always add unit prior
        total_precision = total_precision + 1.0

    if expert_weights is None:
        expert_weights = [torch.ones((batch_size,), dtype=dtype, device=device) for _ in mus]

    for mu, logvar, mask, expert_weight in zip(mus, logvars, masks, expert_weights):
        logvar = clamp_logvar(logvar, logvar_min, logvar_max)
        if expert_weight.ndim != 1 or expert_weight.shape[0] != batch_size:
            raise ValueError(
                "Each expert_weight must have shape [batch_size]; "
                f"got shape={tuple(expert_weight.shape)} for batch_size={batch_size}."
            )
        precision = expert_weight.unsqueeze(-1) * torch.exp(-logvar) * mask.unsqueeze(-1)
        total_precision = total_precision + precision
        weighted_mean = weighted_mean + precision * mu

    fused_var = 1.0 / total_precision.clamp_min(1e-8)
    fused_mu = fused_var * weighted_mean
    fused_logvar = torch.log(fused_var.clamp_min(1e-8))
    # return fused_mu, fused_logvar ## todo: validate clamp_logvar 
    return fused_mu, clamp_logvar(fused_logvar, logvar_min, logvar_max)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool, sample_during_training: bool = True) -> torch.Tensor:
    if not training or not sample_during_training:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
