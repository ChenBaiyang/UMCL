from __future__ import annotations

from pathlib import Path

from models.fusion.artifacts import ArtifactEnsureResult, EnsureArtifactsOptions, ensure_required_artifacts
from models.fusion.config import FusionConfig


def ensure_model_inputs(
    *,
    dataset_id: str,
    mode: str,
    active_views: list[str],
    config: FusionConfig,
    data_root: str | Path = "data",
    input_root: str | Path = "data/preprocessed",
    options: EnsureArtifactsOptions | None = None,
) -> list[ArtifactEnsureResult]:
    return ensure_required_artifacts(
        dataset_id=dataset_id,
        mode=mode,
        active_views=active_views,
        config=config,
        data_root=data_root,
        input_root=input_root,
        options=options,
    )
