from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models.fusion.view_mask import ViewMask


@dataclass
class ModalityPosterior:
    hidden: "torch.Tensor"
    mu: "torch.Tensor"
    logvar: "torch.Tensor"
    mask: "torch.Tensor"


@dataclass
class FusionBatch:
    event_ids: list[str]
    dataset_ids: list[str]
    platforms: list[str]
    canonical_splits: list[str]
    raw_split_sources: list[str | None]
    modalities: dict[str, dict[str, Any]]
    modality_mask: dict[str, "torch.Tensor"]
    task_labels: dict[str, "torch.Tensor"]
    task_mask: dict[str, "torch.Tensor"]
    metadata: list[dict[str, Any]]


@dataclass
class FusionOutput:
    z: "torch.Tensor"
    posterior_mu: "torch.Tensor"
    posterior_logvar: "torch.Tensor"
    modality_posteriors: dict[str, ModalityPosterior]
    reconstruction_params: dict[str, dict[str, "torch.Tensor"]]
    task_outputs: dict[str, dict[str, "torch.Tensor"]]
    auxiliary_outputs: dict[str, "torch.Tensor | None"]
    view_mask: "ViewMask | None" = None
    fusion_variant: str = "poe"
    loss_flags: dict[str, bool] | None = None
