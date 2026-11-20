from __future__ import annotations

from utils.runtime import (
    resolve_device, resolve_amp, resolve_checkpoint_choice, parse_split_arg,
    resolve_optimizer_config as _resolve_optimizer_config, scheduler_params as _scheduler_params,
    now_iso as _now_iso, now_dirstamp as _now_dirstamp,
)

import csv
import argparse
import copy
from pathlib import Path
from typing import Any
import re

import torch

from models.fusion.losses import build_loss_weights



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True, help="Single dataset id, e.g. PHEME")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_root", default="runs")
    parser.add_argument(
        "--pretrained_model",
        "--pretrained_model_name",
        dest="pretrained_model",
        default=None,
        help=(
            "HuggingFace model id or local model path used by the content and stance encoders. "
            "Default: use the configured model (currently xlm-roberta-base)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Global random seed for model init, loaders, and training. Default: 2.",
    )
    parser.add_argument(
        "--kl_loss_weight",
        type=float,
        default=None,
        help="Override the fusion KL loss weight for sweep experiments. Default: use config value.",
    )
    parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=None,
        help="Override the fusion reconstruction loss weight. Default: use config value.",
    )
    parser.add_argument(
        "--task_primary_loss_weight",
        type=float,
        default=None,
        help="Override the fusion primary-task loss weight. Default: use config value.",
    )
    parser.add_argument(
        "--aux_stance_loss_weight",
        type=float,
        default=None,
        help="Override the fusion stance auxiliary loss weight. Default: use config value.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Training/evaluation batch size. Default: use config value.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=None, help="Maximum training epochs. Default: use config value.")
    parser.add_argument("--candidate_epochs", type=int, default=None, help="Number of final epochs to run test and select final best from. Default: use config value.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--checkpoint", default="false", choices=["true", "false"], help="Whether to save the best checkpoint. Default: false.")
    parser.add_argument("--optimizer.backbone.lr", dest="optimizer_backbone_lr", type=float, default=None)
    parser.add_argument("--optimizer.backbone.weight_decay", dest="optimizer_backbone_weight_decay", type=float, default=None)
    parser.add_argument("--optimizer.top_layer.lr", dest="optimizer_top_layer_lr", type=float, default=None)
    parser.add_argument("--optimizer.top_layer.weight_decay", dest="optimizer_top_layer_weight_decay", type=float, default=None)
    parser.add_argument("--optimizer.head.lr", dest="optimizer_head_lr", type=float, default=None)
    parser.add_argument("--optimizer.head.weight_decay", dest="optimizer_head_weight_decay", type=float, default=None)
    parser.add_argument("--splits", default=None, help="Optional comma-separated training splits. Default: train if present, else all.")
    parser.add_argument("--eval_splits", default=None, help="Optional comma-separated evaluation splits. Default: automatic inference (val/dev/test/all/train).")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Override early stopping patience. Default: use config value.",
    )
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--optimizer.scheduler.patience", dest="optimizer_scheduler_patience", type=int, default=None)
    parser.add_argument("--optimizer.scheduler.factor", dest="optimizer_scheduler_factor", type=float, default=None)
    parser.add_argument("--optimizer.scheduler.min_lr", dest="optimizer_scheduler_min_lr", type=float, default=None)
    parser.add_argument("--optimizer.scheduler.start_epoch", dest="optimizer_scheduler_start_epoch", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--active_views",
        default=None,
        help="Comma-separated list of active views for selective training/testing (e.g. 'content,stance'). Default: use config value. Presets: content_only, stance_only, etc.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


















def _sanitize_csv_column_name(raw_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", raw_name).strip("_").lower()
    return sanitized or "class"


def _unique_column_name(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _stringify_active_views(active_views: list[str]) -> str:
    return "|".join(active_views)


def _format_param_count(value: int) -> str:
    return f"{value:,}"


def _print_model_summary(model: torch.nn.Module) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("Model structure:")
    print(model)
    print("Model parameter summary:")
    print(f"  Total params: {_format_param_count(total_params)}")
    print(f"  Trainable params: {_format_param_count(trainable_params)}")
    print(f"  Non-trainable params: {_format_param_count(non_trainable_params)}")
    print("Top-level module parameter breakdown:")
    for module_name, module in model.named_children():
        module_total = sum(param.numel() for param in module.parameters())
        module_trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        print(
            f"  {module_name}: total={_format_param_count(module_total)}, "
            f"trainable={_format_param_count(module_trainable)}"
        )


def _build_core_hyperparameters(
    *,
    args: argparse.Namespace,
    optimizer_config: dict[str, Any],
    config,
    data_root: Path,
    device: torch.device,
    use_amp: bool,
    train_splits: list[str] | None,
    eval_splits: list[str] | None,
    active_views: list[str],
    save_best_checkpoint: bool,
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "pretrained_models": copy.deepcopy(getattr(args, "resolved_pretrained_models", {})),
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "candidate_epochs": args.candidate_epochs,
        "candidate_epoch_start": max(1, int(args.max_epoch) - int(args.candidate_epochs) + 1),
        "candidate_epoch_end": int(args.max_epoch),
        "best_selection_scope": "candidate_epochs",
        "best_selection_rule": "raw_monitor_max_earliest_tie",
        "max_steps": args.max_steps,
        "optimizer": copy.deepcopy(optimizer_config),
        "view_reconstruction_loss_weight": float(config.view_reconstruction_loss_weight),
        "primary_task_loss_weight": float(config.primary_task_loss_weight),
        "stance_auxiliary_loss_weight": float(config.stance_auxiliary_loss_weight),
        "posterior_kl_loss_weight": float(config.posterior_kl_loss_weight),
        "primary_task_loss_type": str(config.primary_task_loss_type),
        "monitor_name": str(config.monitor_name),
        "sample_latent_during_training": bool(config.sample_latent_during_training),
        "loss_weights": build_loss_weights(config),
        "train_splits": train_splits,
        "eval_splits": eval_splits,
        "active_views": active_views,
        "device": str(device),
        "use_amp": use_amp,
        "early_stop_enabled": False,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "save_best_checkpoint": save_best_checkpoint,
    }


def _build_run_config(
    *,
    dataset_id: str,
    mode: str,
    seed: int,
    run_dir: Path,
    start_time: str,
    end_time: str | None,
    core_hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project": "UMCL",
        "run": {
            "dataset_id": dataset_id,
            "mode": mode,
            "seed": seed,
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": end_time,
        },
        "hyperparameters": core_hyperparameters,
    }


def _build_core_summary_hyperparameters(
    *,
    args: argparse.Namespace,
    optimizer_config: dict[str, Any],
    config,
    device: torch.device,
    use_amp: bool,
    active_views: list[str],
    train_splits: list[str] | None,
    eval_splits: list[str] | None,
) -> dict[str, Any]:
    return {
        "pretrained_models": copy.deepcopy(getattr(args, "resolved_pretrained_models", {})),
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "candidate_epochs": args.candidate_epochs,
        "candidate_epoch_start": max(1, int(args.max_epoch) - int(args.candidate_epochs) + 1),
        "candidate_epoch_end": int(args.max_epoch),
        "best_selection_scope": "candidate_epochs",
        "best_selection_rule": "raw_monitor_max_earliest_tie",
        "max_steps": args.max_steps,
        "optimizer": copy.deepcopy(optimizer_config),
        "view_reconstruction_loss_weight": float(config.view_reconstruction_loss_weight),
        "primary_task_loss_weight": float(config.primary_task_loss_weight),
        "stance_auxiliary_loss_weight": float(config.stance_auxiliary_loss_weight),
        "posterior_kl_loss_weight": float(config.posterior_kl_loss_weight),
        "primary_task_loss_type": str(config.primary_task_loss_type),
        "monitor_name": str(config.monitor_name),
        "sample_latent_during_training": bool(config.sample_latent_during_training),
        "loss_weights": build_loss_weights(config),
        "train_splits": train_splits,
        "eval_splits": eval_splits,
        "active_views": active_views,
        "device": str(device),
        "use_amp": use_amp,
        "early_stop": {
            "enabled": False,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
        },
        "save_best_checkpoint": resolve_checkpoint_choice(args.checkpoint),
    }


def _build_eval_log_row(
    *,
    record: Any,
    dataset_id: str,
    seed: int,
    view_reconstruction_loss_weight: float,
    posterior_kl_loss_weight: float,
    active_views: list[str],
    class_names: list[str],
    test_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "epoch": record.epoch,
        "test_macro_f1": "N/A",
        "test_accuracy": "N/A",
        "head_lr": record.head_lr,
        "view_reconstruction_loss_weight": view_reconstruction_loss_weight,
        "posterior_kl_loss_weight": posterior_kl_loss_weight,
        "active_views":_stringify_active_views(active_views),
        "is_best": "true" if record.is_best else "false",
        "best_epoch": record.best_epoch,
        "monitor_name": record.monitor_name,
        "monitor_value": record.monitor_value,
        "best_monitor_value": record.best_monitor_value,
        "early_stop_counter": record.early_stop_counter,
        "train_task_primary_loss": record.train_losses.get("task_primary_loss"),
    }
    for class_name in class_names:
        row[f"test_per_class_f1__{_sanitize_csv_column_name(class_name)}"] = None
    if test_summary is not None:
        metric_name = str(test_summary["metric_name"])
        primary_stats = dict(test_summary["metrics"][metric_name])
        per_class = dict(test_summary["per_class_f1"])
        row["test_macro_f1"] = primary_stats["macro_f1"]
        row["test_accuracy"] = primary_stats["accuracy"]
        for class_name in class_names:
            row[f"test_per_class_f1__{_sanitize_csv_column_name(class_name)}"] = per_class.get(class_name)
    return row


def _eval_log_fieldnames(class_names: list[str]) -> list[str]:
    return [
        "dataset_id",
        "seed",
        "epoch",
        "test_macro_f1",
        "test_accuracy",
        "head_lr",
        "view_reconstruction_loss_weight",
        "posterior_kl_loss_weight",
        "active_views",
        "is_best",
        "best_epoch",
        "monitor_name",
        "monitor_value",
        "best_monitor_value",
        "early_stop_counter",
        "train_task_primary_loss",
        *[f"test_per_class_f1__{_sanitize_csv_column_name(class_name)}" for class_name in class_names]
    ]


def _select_final_best_epoch(records: list[Any], *, test_start_epoch: int) -> tuple[Any | None, list[dict[str, Any]]]:
    candidates = [
        record
        for record in records
        if int(record.epoch) >= int(test_start_epoch)
        and record.monitor_value is not None
    ]
    ranked = sorted(candidates, key=lambda record: (-float(record.monitor_value), int(record.epoch)))
    selected = ranked[0] if ranked else None
    candidate_summary = [
        {
            "epoch": int(record.epoch),
            "monitor_value": float(record.monitor_value),
            "train_task_primary_loss": (
                float(record.train_losses["task_primary_loss"])
                if record.train_losses.get("task_primary_loss") is not None
                else None
            ),
        }
        for record in ranked
    ]
    return selected, candidate_summary


def _append_result_umcl_row(path: Path, fieldnames: list[str], row: dict[str, Any], *, timestamp: str) -> None:
    output_fieldnames = [*fieldnames, "timestamp"]
    output_row = {fieldname: row.get(fieldname) for fieldname in fieldnames}
    output_row["timestamp"] = timestamp
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(output_row)


def _build_summary(
    *,
    dataset_id: str,
    mode: str,
    seed: int,
    run_dir: Path,
    start_time: str,
    end_time: str,
    duration_seconds: float,
    active_views: list[str],
    train_splits: list[str] | None,
    eval_splits: list[str] | None,
    train_sample_count: int,
    eval_sample_count: int,
    feature_dims: dict[str, int],
    primary_task_spec: dict[str, Any],
    core_hyperparameters: dict[str, Any],
    training_result: dict[str, Any],
    last_step: Any,
    last_epoch: Any,
) -> dict[str, Any]:
    return {
        "run": {
            "dataset_id": dataset_id,
            "mode": mode,
            "seed": seed,
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
        },
        "data": {
            "train_splits": train_splits,
            "eval_splits": eval_splits,
            "train_sample_count": train_sample_count,
            "eval_sample_count": eval_sample_count,
            "active_views": active_views,
        },
        "task": {
            "primary_task_spec": primary_task_spec,
            "feature_dims": feature_dims,
        },
        "hyperparameters": core_hyperparameters,
        "training": {
            "epochs_completed": training_result["epochs_completed"],
            "steps_completed": training_result["steps_completed"],
            "stopped_early": training_result["stopped_early"],
            "stop_reason": training_result["stop_reason"],
            "monitor_name": training_result["monitor_name"],
            "best_epoch": training_result["best_epoch"],
            "best_monitor_value": training_result["best_monitor_value"],
            "best_checkpoint": training_result["best_checkpoint"],
            "selection_scope": training_result["selection_scope"],
            "selection_rule": training_result["selection_rule"],
            "candidate_epoch_start": training_result["candidate_epoch_start"],
            "candidate_epoch_end": training_result["candidate_epoch_end"],
            "best_selection_min_delta": training_result["best_selection_min_delta"],
            "cache_stats": training_result["cache_stats"],
        },
        "metrics": {
            "final_step_record": {
                "step": last_step.step,
                "epoch": last_step.epoch,
                "batch_size": last_step.batch_size,
                "dataset_id": last_step.dataset_id,
                "active_views": last_step.active_views,
                "total_loss": last_step.total_loss,
                "reconstruction_loss": last_step.reconstruction_loss,
                "reconstruction_losses": last_step.reconstruction_losses,
                "task_primary_loss": last_step.task_primary_loss,
                "aux_stance_loss": last_step.aux_stance_loss,
                "kl_loss": last_step.kl_loss,
            },
            "final_epoch_record": {
                "epoch": last_epoch.epoch,
                "train_example_count": last_epoch.train_example_count,
                "eval_example_count": last_epoch.eval_example_count,
                "active_views": last_epoch.active_views,
                "train_losses": last_epoch.train_losses,
                "eval_losses": last_epoch.eval_losses,
                "monitor_name": last_epoch.monitor_name,
                "monitor_value": last_epoch.monitor_value,
                "head_lr": last_epoch.head_lr,
                "optimizer_group_lrs": last_epoch.optimizer_group_lrs,
                "is_best": last_epoch.is_best,
                "best_epoch": last_epoch.best_epoch,
                "best_monitor_value": last_epoch.best_monitor_value,
                "early_stop_counter": last_epoch.early_stop_counter,
            },
        },
        "optimizer_final_state": {
            "group_lrs": last_epoch.optimizer_group_lrs,
            "head_lr": last_epoch.head_lr,
        },
    }


def main() -> None:
    from models.training import run_training_cli

    run_training_cli(parse_args())


if __name__ == "__main__":
    main()
