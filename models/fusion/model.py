from __future__ import annotations

import torch
import torch.nn as nn

from config.resolver import resolve_encoder_config
from models.encoders.content_view.models.encoder import ContentEncoder
from models.encoders.evidence_view.models.encoder import EvidenceEncoder
from models.fusion.config import (
    FusionConfig,
    Mode,
    apply_sample_latent_default,
)
from models.fusion.heads import ClassificationHead, GaussianReconstructionHead
from models.fusion.posteriors import product_of_experts, reparameterize
from models.fusion.schemas import FusionBatch, FusionOutput, ModalityPosterior
from models.fusion.view_mask import VIEW_ORDER, ViewMask
from models.encoders.propagation_view.models.encoder import PropagationEncoder
from models.encoders.source_view.models.encoder import SourceEncoder
from models.encoders.stance_view.models.encoder import StanceEncoder


def _scatter_event_level(active_index: torch.Tensor, values: torch.Tensor, batch_size: int) -> torch.Tensor:
    output = values.new_zeros((batch_size, values.shape[-1]))
    if active_index.numel() > 0 and values.numel() > 0:
        output[active_index] = values
    return output


def _get_view_hidden_dim(view_name: str, config_lite: FusionConfig) -> int:
    """Get the hidden dimension for each view using resolved dataset-aware configs."""
    dataset_id = config_lite.dataset_id
    if dataset_id is None:
        raise ValueError("FusionConfig.dataset_id is required for view-dimension resolution.")
    if view_name == "content":
        return resolve_encoder_config("content", dataset_id, config_lite.mode).event_hidden_dim
    if view_name == "stance":
        stance_cfg = resolve_encoder_config("stance", dataset_id, config_lite.mode)
        return stance_cfg.event_hidden_dim
    if view_name == "propagation":
        return resolve_encoder_config("propagation", dataset_id, config_lite.mode).event_hidden_dim
    if view_name == "evidence":
        return resolve_encoder_config("evidence", dataset_id, config_lite.mode).event_hidden_dim
    if view_name == "source":
        return resolve_encoder_config("source", dataset_id, config_lite.mode).event_hidden_dim
    raise ValueError(f"Unknown view: {view_name}")

class UMCLFiveViewModel(nn.Module):
    def __init__(
        self,
        config: FusionConfig,
        feature_dims: dict[str, int],
        primary_task_spec: dict[str, object],
        *,
        active_views: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.primary_task_spec = primary_task_spec
        self.active_views = set(active_views or VIEW_ORDER)
        apply_sample_latent_default(self.config, [view for view in VIEW_ORDER if view in self.active_views])
        mode: Mode = config.mode
        if config.dataset_id is None:
            raise ValueError("FusionConfig.dataset_id is required to initialize UMCLFiveViewModel.")
        self.content_config = resolve_encoder_config("content", config.dataset_id, mode) if "content" in self.active_views else None
        self.stance_config = resolve_encoder_config("stance", config.dataset_id, mode) if "stance" in self.active_views else None
        self.propagation_config = resolve_encoder_config("propagation", config.dataset_id, mode) if "propagation" in self.active_views else None
        self.evidence_config = (
            resolve_encoder_config("evidence", config.dataset_id, mode)
            if "evidence" in self.active_views
            else None
        )
        self.source_config = resolve_encoder_config("source", config.dataset_id, mode) if "source" in self.active_views else None

        if "content" in self.active_views and self.content_config is not None:
            self.content_encoder = ContentEncoder(
                self.content_config,
                mode,
                surface_feature_dim=feature_dims["content_surface"],
                latent_dim=config.posterior_latent_dim,
            )
        else:
            self.content_encoder = None
        if "stance" in self.active_views and self.stance_config is not None:
            stance_reply_embedding_dim = feature_dims.get("stance_reply_embedding")
            if stance_reply_embedding_dim is not None:
                self.stance_config.reply_embedding_dim = int(stance_reply_embedding_dim)
            self.stance_encoder = StanceEncoder(
                self.stance_config,
                mode,
                enable_reply_logits=bool(self.stance_config.enable_reply_classifier_head),
                latent_dim=config.posterior_latent_dim,
            )
            stance_hidden_dim = self.stance_config.event_hidden_dim
        else:
            self.stance_encoder = None
            stance_hidden_dim = _get_view_hidden_dim("stance", self.config)
        if "propagation" in self.active_views and self.propagation_config is not None:
            self.propagation_encoder = PropagationEncoder(
                self.propagation_config,
                mode,
                categorical_feature_dim=feature_dims["propagation_categorical"],
                continuous_feature_dim=feature_dims["propagation_continuous"],
                event_summary_feature_dim=feature_dims.get("propagation_event_summary", 0),
                temporal_profile_feature_dim=feature_dims.get("propagation_temporal_profile", 0),
                enable_node_aux_head=False,
                latent_dim=config.posterior_latent_dim,
            )
        else:
            self.propagation_encoder = None
        if "evidence" in self.active_views and self.evidence_config is not None:
            self.evidence_encoder = EvidenceEncoder(
                self.evidence_config,
                mode,
                summary_feature_dim=feature_dims["evidence_summary"],
                quality_feature_dim=feature_dims["evidence_quality"],
                ref_feature_dim=feature_dims["evidence_ref"],
                latent_dim=config.posterior_latent_dim,
            )
        else:
            self.evidence_encoder = None
        if "source" in self.active_views and self.source_config is not None:
            self.source_encoder = SourceEncoder(
                self.source_config,
                mode,
                summary_feature_dim=feature_dims["source_summary"],
                quality_feature_dim=feature_dims["source_quality"],
                categorical_feature_dim=feature_dims["source_categorical"],
                latent_dim=config.posterior_latent_dim,
            )
        else:
            self.source_encoder = None

        reconstruction_heads: dict[str, GaussianReconstructionHead] = {}
        if "content" in self.active_views and self.content_config is not None:
            reconstruction_heads["content"] = GaussianReconstructionHead(config.posterior_latent_dim, self.content_config.event_hidden_dim, config.fusion_head_hidden_dim, config.fusion_head_dropout)
        if "stance" in self.active_views:
            reconstruction_heads["stance"] = GaussianReconstructionHead(config.posterior_latent_dim, stance_hidden_dim, config.fusion_head_hidden_dim, config.fusion_head_dropout)
        if "propagation" in self.active_views and self.propagation_config is not None:
            reconstruction_heads["propagation"] = GaussianReconstructionHead(config.posterior_latent_dim, self.propagation_config.event_hidden_dim, config.fusion_head_hidden_dim, config.fusion_head_dropout)
        if "evidence" in self.active_views and self.evidence_config is not None:
            reconstruction_heads["evidence"] = GaussianReconstructionHead(config.posterior_latent_dim, self.evidence_config.event_hidden_dim, config.fusion_head_hidden_dim, config.fusion_head_dropout)
        if "source" in self.active_views and self.source_config is not None:
            reconstruction_heads["source"] = GaussianReconstructionHead(config.posterior_latent_dim, self.source_config.event_hidden_dim, config.fusion_head_hidden_dim, config.fusion_head_dropout)
        self.reconstruction_heads = nn.ModuleDict(reconstruction_heads)
        self.task_heads = nn.ModuleDict(
            {
                "primary_task": ClassificationHead(
                    config.posterior_latent_dim,
                    config.fusion_head_hidden_dim,
                    len(self.primary_task_spec["class_names"]),
                    config.fusion_head_dropout,
                ),
            }
        )

    def _create_null_posterior(self, view_name: str, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor]:
        """Create zero posterior for inactive views"""
        batch_size = len(batch.event_ids)
        hidden_dim = _get_view_hidden_dim(view_name, self.config)
        device = batch.modality_mask[view_name].device
        zeros_hidden = torch.zeros((batch_size, hidden_dim), device=device, dtype=torch.float32)
        zeros_latent = torch.zeros((batch_size, self.config.posterior_latent_dim), device=device, dtype=torch.float32)
        
        posterior = ModalityPosterior(
            hidden=zeros_hidden,
            mu=zeros_latent,
            logvar=zeros_latent.clone(),
            mask=batch.modality_mask[view_name],  # Keep original mask
        )
        return posterior, zeros_hidden

    def _encode_content(self, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor]:
        content = batch.modalities["content"]
        output = self.content_encoder(
            input_ids=content["input_ids"],
            attention_mask=content["attention_mask"],
            surface_features=content["surface_features"],
        )
        posterior = ModalityPosterior(
            hidden=output.hidden,
            mu=output.mu,
            logvar=output.logvar,
            mask=batch.modality_mask["content"],
        )
        return posterior, output.hidden

    def _encode_stance(self, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor, torch.Tensor | None]:
        stance = batch.modalities["stance"]
        batch_size = len(batch.event_ids)
        if stance["event_index"].numel() == 0:
            hidden_dim = _get_view_hidden_dim("stance", self.config)
            device = batch.modality_mask["stance"].device
            zeros_hidden = torch.zeros((batch_size, hidden_dim), device=device, dtype=torch.float32)
            zeros_latent = torch.zeros((batch_size, self.config.posterior_latent_dim), device=device, dtype=torch.float32)
            posterior = ModalityPosterior(hidden=zeros_hidden, mu=zeros_latent, logvar=zeros_latent.clone(), mask=batch.modality_mask["stance"])
            return posterior, zeros_hidden, None

        output = self.stance_encoder(
            reply_embedding=stance["reply_embedding"],
            context_cat=stance["context_cat"],
            context_dense=stance["context_dense"],
            event_index=stance["event_index"],
            event_sampled_reply_count=stance["event_sampled_reply_count"],
            event_truncation_flag=stance["event_truncation_flag"],
        )
        active_event_index = torch.unique_consecutive(stance["event_index"])
        hidden = _scatter_event_level(active_event_index, output.event_hidden, batch_size)
        mu = _scatter_event_level(active_event_index, output.mu, batch_size)
        logvar = _scatter_event_level(active_event_index, output.logvar, batch_size)
        posterior = ModalityPosterior(hidden=hidden, mu=mu, logvar=logvar, mask=batch.modality_mask["stance"])
        return posterior, hidden, output.reply_logits

    def _encode_propagation(self, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor]:
        propagation = batch.modalities["propagation"]
        batch_size = len(batch.event_ids)
        if propagation["event_index"].numel() == 0:
            device = batch.modality_mask["propagation"].device
            zeros_hidden = torch.zeros((batch_size, _get_view_hidden_dim("propagation", self.config)), device=device, dtype=torch.float32)
            zeros_latent = torch.zeros((batch_size, self.config.posterior_latent_dim), device=device, dtype=torch.float32)
            posterior = ModalityPosterior(hidden=zeros_hidden, mu=zeros_latent, logvar=zeros_latent.clone(), mask=batch.modality_mask["propagation"])
            return posterior, zeros_hidden

        output = self.propagation_encoder(
            node_features_continuous=propagation["node_features_continuous"],
            node_features_categorical=propagation["node_features_categorical"],
            node_masks=propagation["node_masks"],
            parent_index=propagation["parent_index"],
            event_index=propagation["event_index"],
            event_summary_features=propagation["event_summary_features"],
            temporal_profile_features=propagation["temporal_profile_features"],
        )
        active_event_index = torch.unique_consecutive(propagation["event_index"])
        hidden = _scatter_event_level(active_event_index, output.event_hidden, batch_size)
        mu = _scatter_event_level(active_event_index, output.mu, batch_size)
        logvar = _scatter_event_level(active_event_index, output.logvar, batch_size)
        posterior = ModalityPosterior(hidden=hidden, mu=mu, logvar=logvar, mask=batch.modality_mask["propagation"])
        return posterior, hidden

    def _encode_evidence(self, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor]:
        evidence = batch.modalities["evidence"]
        output = self.evidence_encoder(
            summary_features=evidence["summary_features"],
            summary_quality_flags=evidence["summary_quality_flags"],
            ref_row_features=evidence["ref_row_features"],
            ref_row_mask=evidence["ref_row_mask"],
        )
        posterior = ModalityPosterior(hidden=output.event_hidden, mu=output.mu, logvar=output.logvar, mask=batch.modality_mask["evidence"])
        return posterior, output.event_hidden

    def _encode_source(self, batch: FusionBatch) -> tuple[ModalityPosterior, torch.Tensor]:
        source = batch.modalities["source"]
        output = self.source_encoder(
            summary_features=source["summary_features"],
            summary_quality_flags=source["summary_quality_flags"],
            categorical_features=source["categorical_features"],
        )
        posterior = ModalityPosterior(hidden=output.event_hidden, mu=output.mu, logvar=output.logvar, mask=batch.modality_mask["source"])
        return posterior, output.event_hidden

    def forward(self, batch: FusionBatch, view_mask: ViewMask | None = None) -> FusionOutput:
        """
        Forward pass with optional view masking for selective modality testing
        
        Args:
            batch: FusionBatch containing input data
            view_mask: ViewMask to selectively enable/disable views (None = all views active)
        
        Returns:
            FusionOutput with model outputs
        """
        if view_mask is None:
            view_mask = ViewMask()  # All views active by default
        
        # Conditionally encode each view based on mask
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
        
        # Product-of-Experts fusion with active views only
        active_views = [view for view in view_mask.get_active_views() if view in self.active_views]
        posterior_mu, posterior_logvar = product_of_experts(
            mus=[modality_posteriors[view].mu for view in active_views],
            logvars=[modality_posteriors[view].logvar for view in active_views],
            masks=[modality_posteriors[view].mask for view in active_views],
            use_unit_prior_expert=self.config.use_unit_prior_expert,
            logvar_min=self.config.posterior_logvar_clip_min,
            logvar_max=self.config.posterior_logvar_clip_max,
        )
        z = reparameterize(
            posterior_mu,
            posterior_logvar,
            training=self.training,
            sample_during_training=self.config.sample_latent_during_training,
        )

        targets = {
            "content": content_target,
            "stance": stance_target,
            "propagation": propagation_target,
            "evidence": evidence_target,
            "source": source_target,
        }
        reconstruction_params: dict[str, dict[str, torch.Tensor]] = {}
        for view_name, head in self.reconstruction_heads.items():
            mean, logvar = head(z)
            reconstruction_params[view_name] = {
                "mean": mean,
                "logvar": logvar,
                "target": targets[view_name]#.detach(), # revised 12-Aug-2026 by Baiyang Chen
            }

        task_outputs = {
            "primary_task": {"logits": self.task_heads["primary_task"](z)},
        }
        return FusionOutput(
            z=z,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            modality_posteriors=modality_posteriors,
            reconstruction_params=reconstruction_params,
            task_outputs=task_outputs,
            auxiliary_outputs={"reply_logits": reply_logits},
            view_mask=view_mask,
        )
