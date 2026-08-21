from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config.resolver import resolve_encoder_config
from config.resolver import resolve_fusion_config
from models.fusion.artifacts import EnsureArtifactsOptions, ensure_required_artifacts
from models.fusion.config import (
    apply_optimizer_config,
    normalize_optimizer_config,
)
from models.fusion.evaluation import run_evaluation
from models.fusion.factory import create_fusion_model
from models.fusion.losses import build_loss_weights
from models.fusion.label_space import resolve_primary_task_spec
from models.fusion.training import build_eval_loader
from utils.jsonl import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True, help="Single dataset id, e.g. RumourEval")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained fusion checkpoint.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for test batches, useful for quick checks.")
    parser.add_argument("--splits", default=None, help="Optional comma-separated splits. Default: test, then all, then val/dev, then train.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--output_path", default=None, help="Optional path for test summary JSON.")
    parser.add_argument("--log_path", default=None, help="Optional path for per-batch test log JSONL.")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(choice: str, device: torch.device) -> bool:
    if choice == "true":
        return device.type == "cuda"
    if choice == "false":
        return False
    return device.type == "cuda"


def load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu")


def _attach_primary_metric_fields(summary: dict) -> dict:
    metric_name = str(summary["primary_task"]["metric_name"])
    primary_stats = dict(summary["metrics"][metric_name])
    per_class = primary_stats.get("per_class") or {}
    summary["macro_f1"] = primary_stats.get("macro_f1")
    summary["accuracy"] = primary_stats.get("accuracy")
    summary["per_class_f1"] = {
        class_name: stats.get("f1")
        for class_name, stats in per_class.items()
    }
    summary["metric_name"] = metric_name
    return summary


def build_test_loader_with_fallback(
    *,
    dataset_id: str,
    config,
    data_root: Path,
    batch_size: int,
    device: torch.device,
    num_workers: int | None,
    prefetch_factor: int | None,
    disable_persistent_workers: bool,
    disable_pin_memory: bool,
    required_views: set[str] | None = None,
):
    candidate_splits = [["test"], ["all"], ["val"], ["dev"], ["train"]]
    last_error: ValueError | None = None
    for candidate in candidate_splits:
        try:
            return build_eval_loader(
                dataset_id=dataset_id,
                config=config,
                data_root=data_root,
                batch_size=batch_size,
                splits=candidate,
                device=device,
                num_workers_override=num_workers,
                prefetch_factor_override=prefetch_factor,
                persistent_workers_override=(False if disable_persistent_workers else None),
                pin_memory_override=(False if disable_pin_memory else None),
                required_views=required_views,
            )
        except ValueError as exc:
            if "No evaluation samples found" not in str(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"No evaluation samples found for dataset={dataset_id}.")


def run_test_from_model(
    *,
    model,
    dataset_id: str,
    mode: str,
    config,
    primary_task_spec: dict[str, Any],
    loader,
    sample_count: int,
    selected_splits: list[str],
    feature_dims: dict[str, int],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    active_views: list[str],
    checkpoint_path: Path | None = None,
    max_batches: int | None = None,
    output_path: Path | None = None,
    log_path: Path | None = None,
    loader_settings: dict[str, Any] | None = None,
    loader_runtime_overrides: dict[str, Any] | None = None,
    stance_config=None,
    propagation_config=None,
) -> dict:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    summary = run_evaluation(
        model=model,
        loader=loader,
        config=config,
        primary_task_spec=primary_task_spec,
        device=device,
        log_path=log_path,
        max_batches=max_batches,
        use_amp=use_amp,
    )
    summary.update(
        {
            "checkpoint": (str(checkpoint_path) if checkpoint_path is not None else None),
            "mode": mode,
            "fusion_variant": str(getattr(config, "fusion_variant", "poe") or "poe"),
            "active_views": active_views,
            "batch_size": batch_size,
            "max_batches_requested": max_batches,
            "device": str(device),
            "use_amp": use_amp,
            "splits": selected_splits,
            "sample_count": sample_count,
            "feature_dims": feature_dims,
            "primary_task_spec": primary_task_spec,
            "log_path": (str(log_path) if log_path is not None else None),
            "loader_settings": loader_settings,
            "loader_runtime_overrides": loader_runtime_overrides,
            "stance_max_replies_per_event": stance_config.max_replies_per_event_value() if stance_config is not None else None,
            "propagation_max_nodes_per_event": propagation_config.max_nodes_per_event_value() if propagation_config is not None else None,
            "view_reconstruction_loss_weight": float(config.view_reconstruction_loss_weight),
            "primary_task_loss_weight": float(config.primary_task_loss_weight),
            "stance_auxiliary_loss_weight": float(config.stance_auxiliary_loss_weight),
            "posterior_kl_loss_weight": float(config.posterior_kl_loss_weight),
            "loss_weights": build_loss_weights(config),
        }
    )
    summary = _attach_primary_metric_fields(summary)
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def run_test(
    *,
    dataset_id: str,
    checkpoint_path: Path,
    mode: str | None,
    data_root: Path,
    batch_size: int,
    num_workers: int | None,
    prefetch_factor: int | None,
    disable_persistent_workers: bool,
    disable_pin_memory: bool,
    max_batches: int | None,
    splits: list[str] | None,
    device: torch.device,
    use_amp: bool,
    output_path: Path | None = None,
    log_path: Path | None = None,
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_cfg_payload = dict(checkpoint["fusion_config"])
    checkpoint_mode = checkpoint_cfg_payload.get("mode", "full")
    mode = mode or checkpoint_mode
    active_views = list(checkpoint.get("active_views") or ["content", "stance", "propagation", "evidence", "source"])
    checkpoint_runtime_overrides = {
        key: value
        for key, value in checkpoint_cfg_payload.items()
        if key not in {"dataset_id", "mode", "resolved_overrides"}
    }
    config = resolve_fusion_config(
        dataset_id=dataset_id,
        mode=mode,
    )
    config = apply_optimizer_config(config, normalize_optimizer_config(config.optimizer))
    stance_config = resolve_encoder_config("stance", dataset_id, mode)
    propagation_config = resolve_encoder_config("propagation", dataset_id, mode)
    for key, value in checkpoint_runtime_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config = apply_optimizer_config(config, normalize_optimizer_config(config.optimizer))
    checkpoint_primary_task_spec = dict(checkpoint["primary_task_spec"])
    dataset_primary_task_spec = resolve_primary_task_spec(dataset_id, data_root)
    if checkpoint_primary_task_spec["task_name"] != dataset_primary_task_spec["task_name"]:
        raise ValueError("Checkpoint primary task name does not match dataset primary task.")
    if list(checkpoint_primary_task_spec["class_names"]) != list(dataset_primary_task_spec["class_names"]):
        raise ValueError("Checkpoint primary task class names do not match dataset primary task.")

    if splits is None:
        ensure_required_artifacts(
            dataset_id=dataset_id,
            mode=mode,
            active_views=active_views,
            config=config,
            data_root=data_root,
            input_root=data_root / "preprocessed",
            options=EnsureArtifactsOptions(),
        )
        loader, sample_count, selected_splits, feature_dims, loader_settings, primary_task_spec = build_test_loader_with_fallback(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            disable_persistent_workers=disable_persistent_workers,
            disable_pin_memory=disable_pin_memory,
            required_views=set(active_views),
        )
    else:
        ensure_required_artifacts(
            dataset_id=dataset_id,
            mode=mode,
            active_views=active_views,
            config=config,
            data_root=data_root,
            input_root=data_root / "preprocessed",
            options=EnsureArtifactsOptions(),
        )
        loader, sample_count, selected_splits, feature_dims, loader_settings, primary_task_spec = build_eval_loader(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=batch_size,
            splits=splits,
            device=device,
            num_workers_override=num_workers,
            prefetch_factor_override=prefetch_factor,
            persistent_workers_override=(False if disable_persistent_workers else None),
            pin_memory_override=(False if disable_pin_memory else None),
            required_views=set(active_views),
        )

    model = create_fusion_model(
        config,
        feature_dims,
        primary_task_spec=primary_task_spec,
        active_views=set(active_views),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    summary = run_test_from_model(
        model=model,
        dataset_id=dataset_id,
        mode=mode,
        config=config,
        primary_task_spec=primary_task_spec,
        loader=loader,
        sample_count=sample_count,
        selected_splits=selected_splits,
        feature_dims=feature_dims,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        active_views=active_views,
        checkpoint_path=checkpoint_path,
        max_batches=max_batches,
        output_path=output_path,
        log_path=log_path,
        loader_settings=loader_settings,
        loader_runtime_overrides={
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "disable_persistent_workers": disable_persistent_workers,
            "disable_pin_memory": disable_pin_memory,
        },
        stance_config=stance_config,
        propagation_config=propagation_config,
    )
    return summary


def main() -> None:
    from models.evaluation import run_evaluation_cli

    summary = run_evaluation_cli(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
