from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

Mode = Literal["lite", "full"]
SUPPORTED_MONITOR_NAMES = ("macro_f1", "accuracy", "composite")


def _default_optimizer_config() -> dict[str, Any]:
    return {
        "backbone": {"lr": None, "weight_decay": None},
        "top_layer": {"lr": None, "weight_decay": None},
        "head": {"lr": None, "weight_decay": None},
        "scheduler": {"factor": None, "patience": None, "min_lr": None, "start_epoch": None},
    }


@dataclass
class FusionConfig:
    dataset_id: str | None = None
    mode: Mode = "full"
    fusion_variant: str = "poe"
    posterior_latent_dim: int = 256
    posterior_logvar_clip_min: float = -8.0
    posterior_logvar_clip_max: float = 8.0
    sample_latent_during_training: bool | None = None
    use_unit_prior_expert: bool = True
    fusion_head_hidden_dim: int = 256
    fusion_head_dropout: float = 0.1

    view_reconstruction_loss_weight: float = 0.1
    primary_task_loss_weight: float = 1.0
    stance_auxiliary_loss_weight: float = 0.0
    posterior_kl_loss_weight: float = 1e-5
    primary_task_loss_type: str = "cross_entropy"

    views: list[str] = field(default_factory=lambda: ["content", "stance", "propagation", "evidence", "source"])
    active_views: str = "content,propagation,stance"
    batch_size: int = 32
    max_auto_workers: int = 4
    data_loader_seed: int = 13
    max_open_shards: int = 16
    last_checkpoint_interval: int = 1
    early_stop_patience: int = 5
    max_epoch: int = 30
    candidate_epochs: int = 10
    monitor_name: str = "macro_f1"
    optimizer: dict[str, Any] = field(default_factory=_default_optimizer_config)
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def bucket_batching_enabled(self) -> bool:
        return False

    def stance_cached_embeddings_enabled(self) -> bool:
        return False

    def stance_backbone_sharing_enabled(self) -> bool:
        return False

    def max_open_shards_limit(self) -> int:
        return int(self.max_open_shards)

    def last_checkpoint_interval_epochs(self) -> int:
        return int(self.last_checkpoint_interval)

    def artifact_mode(self) -> Literal["lite", "full"]:
        return "full" if self.mode == "full" else "lite"

    def as_dict(self) -> dict:
        return asdict(self)


def validate_monitor_name(value: str) -> str:
    monitor_name = str(value or "macro_f1").strip().lower()
    if monitor_name not in SUPPORTED_MONITOR_NAMES:
        raise ValueError(
            f"Unsupported monitor_name: {monitor_name}. "
            f"Supported values: {', '.join(SUPPORTED_MONITOR_NAMES)}"
        )
    return monitor_name


def infer_sample_latent_during_training(active_views: Sequence[str] | None, all_views: Sequence[str] | None = None) -> bool:
    views = [str(view) for view in (active_views or all_views or []) if str(view)]
    return len(views) > 1


def apply_sample_latent_default(config: FusionConfig, active_views: Sequence[str] | None = None) -> FusionConfig:
    if config.sample_latent_during_training is None:
        configured_views = [view.strip() for view in str(config.active_views or "").replace(";", ",").split(",") if view.strip()]
        if not configured_views or str(config.active_views).strip().lower() == "all":
            configured_views = list(config.views)
        config.sample_latent_during_training = infer_sample_latent_during_training(
            active_views if active_views is not None else configured_views,
            config.views,
        )
    return config


def normalize_optimizer_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    normalized = copy.deepcopy(_default_optimizer_config())
    incoming = raw or {}
    for group_name in ("backbone", "top_layer", "head", "scheduler"):
        group_payload = incoming.get(group_name)
        if not isinstance(group_payload, dict):
            continue
        for key, value in group_payload.items():
            if key in normalized[group_name]:
                normalized[group_name][key] = copy.deepcopy(value)
    return normalized


def apply_optimizer_config(config: FusionConfig, optimizer_config: dict[str, Any]) -> FusionConfig:
    config.optimizer = normalize_optimizer_config(optimizer_config)
    config.resolved_overrides = {
        **config.resolved_overrides,
        "optimizer": copy.deepcopy(config.optimizer),
    }
    return config
