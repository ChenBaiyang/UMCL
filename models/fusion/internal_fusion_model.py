from __future__ import annotations

import torch
import torch.nn as nn

from models.fusion.heads import ClassificationHead
from models.fusion.model import UMCLFiveViewModel, _get_view_hidden_dim
from models.fusion.posteriors import reparameterize
from models.fusion.schemas import FusionBatch, FusionOutput
from models.fusion.view_mask import VIEW_ORDER, ViewMask


SUPPORTED_INTERNAL_FUSION_VARIANTS = {"concat", "gated", "mean_posterior"}


def _masked_mean(stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=stack.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (stack * weights).sum(dim=1) / denom


class InternalFusionBaselineModel(UMCLFiveViewModel):
    def __init__(
        self,
        config,
        feature_dims: dict[str, int],
        primary_task_spec: dict[str, object],
        *,
        active_views: set[str] | None = None,
        fusion_variant: str,
    ) -> None:
        normalized_variant = str(fusion_variant or "").strip().lower()
        if normalized_variant not in SUPPORTED_INTERNAL_FUSION_VARIANTS:
            raise ValueError(
                f"Unsupported internal fusion variant: {fusion_variant}. "
                f"Expected one of: {sorted(SUPPORTED_INTERNAL_FUSION_VARIANTS)}"
            )
        config.fusion_variant = normalized_variant
        super().__init__(
            config,
            feature_dims,
            primary_task_spec=primary_task_spec,
            active_views=active_views,
        )
        self.fusion_variant = normalized_variant
        self.active_view_order = [view for view in VIEW_ORDER if view in self.active_views]

        projection_layers: dict[str, nn.Module] = {}
        gate_layers: dict[str, nn.Module] = {}
        for view_name in self.active_view_order:
            hidden_dim = _get_view_hidden_dim(view_name, self.config)
            projection_layers[view_name] = nn.Linear(hidden_dim, self.config.posterior_latent_dim)
            if self.fusion_variant == "gated":
                gate_layers[view_name] = nn.Linear(self.config.posterior_latent_dim, 1)
        self.view_projection_layers = nn.ModuleDict(projection_layers)
        self.view_gate_layers = nn.ModuleDict(gate_layers)

        class_count = len(self.primary_task_spec["class_names"])
        if self.fusion_variant == "concat":
            classifier_input_dim = self.config.posterior_latent_dim * len(self.active_view_order)
        else:
            classifier_input_dim = self.config.posterior_latent_dim
        self.task_heads["primary_task"] = ClassificationHead(
            classifier_input_dim,
            self.config.fusion_head_hidden_dim,
            class_count,
            self.config.fusion_head_dropout,
        )
        if self.fusion_variant in {"concat", "gated"}:
            self.reconstruction_heads = nn.ModuleDict({})

    def _encode_all_views(
        self,
        batch: FusionBatch,
        view_mask: ViewMask,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor], torch.Tensor | None]:
        content_posterior, content_target = (
            self._encode_content(batch) if view_mask.is_active("content") and "content" in self.active_views
            else self._create_null_posterior("content", batch)
        )
        stance_posterior, stance_target, reply_logits = (
            self._encode_stance(batch) if view_mask.is_active("stance") and "stance" in self.active_views
            else (*self._create_null_posterior("stance", batch), None)
        )
        propagation_posterior, propagation_target = (
            self._encode_propagation(batch) if view_mask.is_active("propagation") and "propagation" in self.active_views
            else self._create_null_posterior("propagation", batch)
        )
        evidence_posterior, evidence_target = (
            self._encode_evidence(batch) if view_mask.is_active("evidence") and "evidence" in self.active_views
            else self._create_null_posterior("evidence", batch)
        )
        source_posterior, source_target = (
            self._encode_source(batch) if view_mask.is_active("source") and "source" in self.active_views
            else self._create_null_posterior("source", batch)
        )
        modality_posteriors = {
            "content": content_posterior,
            "stance": stance_posterior,
            "propagation": propagation_posterior,
            "evidence": evidence_posterior,
            "source": source_posterior,
        }
        targets = {
            "content": content_target,
            "stance": stance_target,
            "propagation": propagation_target,
            "evidence": evidence_target,
            "source": source_target,
        }
        return modality_posteriors, targets, reply_logits

    def _projected_view_stack(
        self,
        modality_posteriors: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_views: list[torch.Tensor] = []
        mask_views: list[torch.Tensor] = []
        for view_name in self.active_view_order:
            posterior = modality_posteriors[view_name]
            projected = self.view_projection_layers[view_name](posterior.hidden)
            projected = projected * posterior.mask.unsqueeze(-1).to(dtype=projected.dtype)
            projected_views.append(projected)
            mask_views.append(posterior.mask)
        return torch.stack(projected_views, dim=1), torch.stack(mask_views, dim=1)

    def _forward_concat_or_gated(
        self,
        modality_posteriors: dict[str, object],
        reply_logits: torch.Tensor | None,
        view_mask: ViewMask,
    ) -> FusionOutput:
        projected_stack, availability_mask = self._projected_view_stack(modality_posteriors)
        if self.fusion_variant == "concat":
            z = projected_stack.reshape(projected_stack.shape[0], -1)
        else:
            gate_logits = torch.cat(
                [self.view_gate_layers[view_name](projected_stack[:, idx, :]) for idx, view_name in enumerate(self.active_view_order)],
                dim=1,
            )
            gate_logits = gate_logits.masked_fill(availability_mask <= 0, -1e9)
            all_missing = availability_mask.sum(dim=1) <= 0
            if torch.any(all_missing):
                gate_logits = gate_logits.clone()
                gate_logits[all_missing] = 0.0
            gate_weights = torch.softmax(gate_logits, dim=1)
            gate_weights = gate_weights * availability_mask.to(dtype=projected_stack.dtype)
            gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            z = (projected_stack * gate_weights.unsqueeze(-1)).sum(dim=1)

        batch_size = z.shape[0]
        zeros = z.new_zeros((batch_size, self.config.posterior_latent_dim))
        return FusionOutput(
            z=z,
            posterior_mu=zeros,
            posterior_logvar=zeros,
            modality_posteriors=modality_posteriors,
            reconstruction_params={},
            task_outputs={"primary_task": {"logits": self.task_heads["primary_task"](z)}},
            auxiliary_outputs={"reply_logits": reply_logits},
            view_mask=view_mask,
            fusion_variant=self.fusion_variant,
            loss_flags={"reconstruction": False, "aux_stance": False, "kl": False},
        )

    def _forward_mean_posterior(
        self,
        modality_posteriors: dict[str, object],
        targets: dict[str, torch.Tensor],
        reply_logits: torch.Tensor | None,
        view_mask: ViewMask,
    ) -> FusionOutput:
        mu_stack = torch.stack([modality_posteriors[view].mu for view in self.active_view_order], dim=1)
        logvar_stack = torch.stack([modality_posteriors[view].logvar for view in self.active_view_order], dim=1)
        mask_stack = torch.stack([modality_posteriors[view].mask for view in self.active_view_order], dim=1)
        posterior_mu = _masked_mean(mu_stack, mask_stack)
        posterior_logvar = torch.clamp(
            _masked_mean(logvar_stack, mask_stack),
            min=self.config.posterior_logvar_clip_min,
            max=self.config.posterior_logvar_clip_max,
        )
        z = reparameterize(
            posterior_mu,
            posterior_logvar,
            training=self.training,
            sample_during_training=self.config.sample_latent_during_training,
        )
        reconstruction_params: dict[str, dict[str, torch.Tensor]] = {}
        for view_name, head in self.reconstruction_heads.items():
            mean, logvar = head(z)
            reconstruction_params[view_name] = {"mean": mean, "logvar": logvar, "target": targets[view_name]}
        return FusionOutput(
            z=z,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            modality_posteriors=modality_posteriors,
            reconstruction_params=reconstruction_params,
            task_outputs={"primary_task": {"logits": self.task_heads["primary_task"](z)}},
            auxiliary_outputs={"reply_logits": reply_logits},
            view_mask=view_mask,
            fusion_variant=self.fusion_variant,
            loss_flags={"reconstruction": True, "aux_stance": True, "kl": True},
        )

    def forward(self, batch: FusionBatch, view_mask: ViewMask | None = None) -> FusionOutput:
        if view_mask is None:
            view_mask = ViewMask()
        modality_posteriors, targets, reply_logits = self._encode_all_views(batch, view_mask)
        if self.fusion_variant in {"concat", "gated"}:
            return self._forward_concat_or_gated(modality_posteriors, reply_logits, view_mask)
        return self._forward_mean_posterior(modality_posteriors, targets, reply_logits, view_mask)
