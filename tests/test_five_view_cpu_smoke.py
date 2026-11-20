from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from config.resolver import resolve_encoder_config, resolve_fusion_config
from models.fusion.losses import build_loss_weights, compute_total_loss
from models.fusion.model import UMCLFiveViewModel
from models.fusion.posteriors import product_of_experts
from models.fusion.schemas import FusionBatch
from utils.primary_label_space import resolve_primary_task_spec


class _LocalBackbone(nn.Module):
    """Local embedding backbone avoids downloading pretrained weights in smoke tests."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("dataset", ["PHEME", "Twitter15", "Twitter16", "Weibo"])
def test_five_views_loss_and_strict_checkpoint_roundtrip(dataset):
    torch.manual_seed(2)
    config = resolve_fusion_config(dataset)
    config.posterior_latent_dim = 8
    config.fusion_head_hidden_dim = 16
    configs = {v: resolve_encoder_config(v, dataset) for v in
               ["content", "stance", "propagation", "evidence", "source"]}
    for c in configs.values():
        c.event_hidden_dim = 16
    configs["propagation"].node_hidden_dim = 16
    configs["propagation"].transformer_layer_count = 1
    configs["propagation"].transformer_ffn_hidden_dim = 32
    configs["stance"].reply_embedding_dim = 16
    dims = {"content_surface": 3, "stance_reply_embedding": 16,
            "propagation_categorical": 14, "propagation_continuous": 8,
            "propagation_event_summary": 23, "propagation_temporal_profile": 38,
            "evidence_summary": 3, "evidence_quality": 2, "evidence_ref": 8,
            "source_summary": 3, "source_quality": 2, "source_categorical": 4}
    spec = resolve_primary_task_spec(dataset)
    # Both events contain two ordered replies and a three-node tree.
    modalities = {
        "content": {"input_ids": torch.ones(2, 4, dtype=torch.long),
                    "attention_mask": torch.ones(2, 4, dtype=torch.long),
                    "surface_features": torch.randn(2, 3)},
        "stance": {"reply_embedding": torch.randn(4, 16),
                   "context_cat": torch.ones(4, 4, dtype=torch.long),
                   "context_dense": torch.zeros(4, 17),
                   "event_index": torch.tensor([0, 0, 1, 1]),
                   "event_sampled_reply_count": torch.full((4,), 2.),
                   "event_truncation_flag": torch.zeros(4),
                   "reply_label_ids": torch.full((4,), -1, dtype=torch.long),
                   "reply_label_mask": torch.zeros(4)},
        "propagation": {"node_features_continuous": torch.randn(6, 8),
                        "node_features_categorical": torch.ones(6, 14, dtype=torch.long),
                        "node_masks": torch.ones(6),
                        "parent_index": torch.tensor([-1, 0, 1, -1, 0, 1]),
                        "event_index": torch.tensor([0, 0, 0, 1, 1, 1]),
                        "event_summary_features": torch.randn(2, 23),
                        "temporal_profile_features": torch.randn(2, 38)},
        "evidence": {"summary_features": torch.randn(2, 3),
                     "summary_quality_flags": torch.zeros(2, 2),
                     "ref_row_features": torch.zeros(2, 1, 8, dtype=torch.long),
                     "ref_row_mask": torch.ones(2, 1)},
        "source": {"summary_features": torch.randn(2, 3),
                   "summary_quality_flags": torch.zeros(2, 2),
                   "categorical_features": torch.zeros(2, 4, dtype=torch.long)},
    }
    batch = FusionBatch(event_ids=["a", "b"], dataset_ids=[dataset] * 2,
                        platforms=["twitter"] * 2, canonical_splits=["train"] * 2,
                        raw_split_sources=[None] * 2, modalities=modalities,
                        modality_mask={v: torch.ones(2) for v in configs},
                        task_labels={"primary_task": torch.tensor([0, 1])},
                        task_mask={"primary_task": torch.ones(2)}, metadata=[{}, {}])
    with patch("models.fusion.model.resolve_encoder_config", side_effect=lambda v, *args: configs[v]), \
         patch("models.encoders.content_view.models.encoder.load_backbone",
               side_effect=lambda *args: (_LocalBackbone(), 16)):
        model = UMCLFiveViewModel(config, dims, spec).eval()
        assert model.propagation_encoder.config.parent_bias_attention_enabled
        output = model(batch)
        assert output.task_outputs["primary_task"]["logits"].shape == (2, len(spec["class_names"]))
        assert set(output.reconstruction_params) == set(configs)
        losses = compute_total_loss(output, batch, build_loss_weights(config))
        assert torch.isfinite(losses["total_loss"])
        assert losses["aux_stance_loss"].item() == 0
        losses["total_loss"].backward()
        assert model.propagation_encoder.transformer.layers[0].attn.q_proj.weight.grad is not None
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = UMCLFiveViewModel(config, dims, spec).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu"), strict=True)
        torch.testing.assert_close(restored(batch).posterior_mu, output.posterior_mu)
        # Event-level missingness must exclude that expert from PoE.
        batch.modality_mask["propagation"][1] = 0
        masked = model(batch)
        views = list(configs)
        mu, _ = product_of_experts(
            [masked.modality_posteriors[v].mu for v in views],
            [masked.modality_posteriors[v].logvar for v in views],
            [masked.modality_posteriors[v].mask for v in views],
            use_unit_prior_expert=config.use_unit_prior_expert,
            logvar_min=config.posterior_logvar_clip_min, logvar_max=config.posterior_logvar_clip_max)
        torch.testing.assert_close(masked.posterior_mu, mu)
