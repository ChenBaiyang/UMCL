from __future__ import annotations

from models.fusion.internal_fusion_model import InternalFusionBaselineModel
from models.fusion.model import UMCLFiveViewModel


def create_fusion_model(
    config,
    feature_dims: dict[str, int],
    *,
    primary_task_spec: dict[str, object],
    active_views: set[str] | None = None,
):
    fusion_variant = str(getattr(config, "fusion_variant", "poe") or "poe").strip().lower()
    if fusion_variant == "poe":
        return UMCLFiveViewModel(
            config,
            feature_dims,
            primary_task_spec=primary_task_spec,
            active_views=active_views,
        )
    return InternalFusionBaselineModel(
        config,
        feature_dims,
        primary_task_spec=primary_task_spec,
        active_views=active_views,
        fusion_variant=fusion_variant,
    )
