"""UMCL experiment: missing views."""

from __future__ import annotations

from utils.runtime import move_batch_to_device as _move_batch_to_device, autocast_context as _autocast_context

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from config.resolver import resolve_fusion_config
from models.artifacts import EnsureArtifactsOptions, ensure_model_inputs
from models.evaluation import build_test_loader_with_fallback
from models.fusion.config import (
    apply_optimizer_config,
    apply_sample_latent_default,
    normalize_optimizer_config,
)
from models.fusion.factory import create_fusion_model
from models.fusion.hierarchical_optimizer import build_hierarchical_optimizer
from models.fusion.training import build_eval_loader, build_train_loader, run_training, save_checkpoint
from models.fusion.view_mask import ViewMask
from train_umcl import (
    _now_dirstamp,
    _now_iso,
    _resolve_optimizer_config,
    _scheduler_params,
    parse_split_arg,
    resolve_amp,
    resolve_checkpoint_choice,
    resolve_device,
)
from utils.jsonl import write_json
from utils.torch_runtime import seed_everything


FULL_VIEWS = ("content", "stance", "propagation")
DEFAULT_DATASETS = ("PHEME", "Weibo")
DEFAULT_VARIANTS = ("concat", "gated", "mean_posterior", "poe")
DEFAULT_SEEDS = (2, 4, 8, 16, 32)
DIAGNOSTIC_VERSION = 1

MASK_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("complete", ("content", "stance", "propagation"), "Complete"),
    ("content_stance", ("content", "stance"), "C/S"),
    ("content_propagation", ("content", "propagation"), "C/P"),
    ("stance_propagation", ("stance", "propagation"), "S/P"),
    ("content_only", ("content",), "C"),
    ("stance_only", ("stance",), "S"),
    ("propagation_only", ("propagation",), "P"),
)
CONDITION_MASKS = {
    "complete": ("complete",),
    "one_missing": ("content_stance", "content_propagation", "stance_propagation"),
    "two_missing": ("content_only", "stance_only", "propagation_only"),
}
CONDITION_LABELS = {
    "complete": "Complete",
    "one_missing": "One missing",
    "two_missing": "Two missing",
}
VARIANT_LABELS = {
    "concat": "MLP Posterior Fusion",
    "gated": "Gated Posterior Fusion",
    "mean_posterior": "Mean Posterior Fusion",
    "poe": "PoE Fusion",
}


def _comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _seed_list(raw: str) -> list[int]:
    values = [int(item) for item in _comma_list(raw)]
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train complete C/S/P fusion models and evaluate missing-view robustness."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--fusion_variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", type=_seed_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_root", default="runs_missing_view_robustness")
    parser.add_argument("--force", action="store_true", help="Rerun matching completed experiments.")
    parser.add_argument("--kl_loss_weight", type=float, default=1e-5)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=1e-2)
    parser.add_argument("--task_primary_loss_weight", type=float, default=1)
    parser.add_argument("--aux_stance_loss_weight", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--candidate_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--checkpoint", default="false", choices=["true", "false"])
    parser.add_argument("--optimizer.backbone.lr", dest="optimizer_backbone_lr", type=float, default=None)
    parser.add_argument(
        "--optimizer.backbone.weight_decay", dest="optimizer_backbone_weight_decay", type=float, default=None
    )
    parser.add_argument("--optimizer.top_layer.lr", dest="optimizer_top_layer_lr", type=float, default=None)
    parser.add_argument(
        "--optimizer.top_layer.weight_decay", dest="optimizer_top_layer_weight_decay", type=float, default=None
    )
    parser.add_argument("--optimizer.head.lr", dest="optimizer_head_lr", type=float, default=None)
    parser.add_argument(
        "--optimizer.head.weight_decay", dest="optimizer_head_weight_decay", type=float, default=None
    )
    parser.add_argument("--splits", default=None)
    parser.add_argument("--eval_splits", default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer.scheduler.patience", dest="optimizer_scheduler_patience", type=int, default=None
    )
    parser.add_argument("--optimizer.scheduler.factor", dest="optimizer_scheduler_factor", type=float, default=None)
    parser.add_argument("--optimizer.scheduler.min_lr", dest="optimizer_scheduler_min_lr", type=float, default=None)
    parser.add_argument(
        "--optimizer.scheduler.start_epoch", dest="optimizer_scheduler_start_epoch", type=int, default=None
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    return parser.parse_args()


def _validate_grid(args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    datasets = _comma_list(args.datasets)
    variants = [item.lower() for item in _comma_list(args.fusion_variants)]
    seeds = list(args.seeds)
    if not datasets:
        raise ValueError("At least one dataset is required.")
    invalid_variants = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if invalid_variants:
        raise ValueError(f"Unsupported fusion variants: {invalid_variants}; expected {list(DEFAULT_VARIANTS)}")
    if len(variants) != len(set(variants)):
        raise ValueError("fusion_variants must not contain duplicates.")
    if len(datasets) != len(set(datasets)):
        raise ValueError("datasets must not contain duplicates.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must not contain duplicates.")
    return datasets, variants, seeds


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_completed_run(
    run_root: Path, fusion_variant: str, dataset_id: str, seed: int, experiment_signature: str
) -> dict[str, Any] | None:
    base = run_root / "runs" / fusion_variant / dataset_id / f"seed_{seed}"
    if not base.exists():
        return None
    for path in sorted(base.glob("*/summary.json"), reverse=True):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "complete"
            and payload.get("experiment_signature") == experiment_signature
            and len(payload.get("masked_tests") or {}) == len(MASK_SPECS)
        ):
            return payload
    return None






def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, class_count: int) -> float:
    valid = valid_mask > 0
    if not torch.any(valid):
        raise ValueError("Masked test produced no valid primary-task labels.")
    y_true = labels[valid].long().view(-1)
    y_pred = logits[valid].argmax(dim=-1).long().view(-1)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    for gold, predicted in zip(y_true, y_pred):
        confusion[int(gold.item()), int(predicted.item())] += 1
    f1_values: list[float] = []
    for class_id in range(class_count):
        tp = float(confusion[class_id, class_id].item())
        fp = float(confusion[:, class_id].sum().item()) - tp
        fn = float(confusion[class_id, :].sum().item()) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(sum(f1_values) / class_count)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask > 0
    if not torch.any(valid):
        raise ValueError("Masked test produced no valid primary-task labels.")
    predictions = logits[valid].argmax(dim=-1)
    return float(predictions.eq(labels[valid]).float().mean().item())


def _uncertainty_group(values: torch.Tensor) -> dict[str, float | int | None]:
    count = int(values.numel())
    return {"count": count, "mean": (float(values.double().mean().item()) if count else None)}


@torch.no_grad()
def _run_masked_test(
    *,
    model: torch.nn.Module,
    loader: Any,
    primary_task_spec: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    view_mask: ViewMask,
    collect_uncertainty: bool,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    uncertainty_parts: list[torch.Tensor] = []
    event_count = 0
    non_blocking = device.type == "cuda"

    for batch in loader:
        _move_batch_to_device(batch, device, non_blocking=non_blocking)
        with _autocast_context(device, use_amp):
            output = model(batch, view_mask=view_mask)
        logits = output.task_outputs["primary_task"]["logits"].detach().float().cpu()
        labels = batch.task_labels["primary_task"].detach().cpu()
        valid_mask = batch.task_mask["primary_task"].detach().cpu()
        logits_parts.append(logits)
        label_parts.append(labels)
        valid_parts.append(valid_mask)
        event_count += len(batch.event_ids)
        if collect_uncertainty:
            if str(getattr(output, "fusion_variant", "poe")).lower() != "poe":
                raise ValueError("Posterior uncertainty is only defined for the PoE/UMCL output.")
            posterior_logvar = output.posterior_logvar.detach().float()
            if posterior_logvar.ndim != 2 or posterior_logvar.shape[0] != logits.shape[0]:
                raise ValueError("Unexpected fused posterior_logvar shape during uncertainty evaluation.")
            uncertainty_parts.append(posterior_logvar.exp().mean(dim=-1).cpu())

    if not logits_parts:
        raise ValueError("Masked test loader produced zero batches.")
    logits_all = torch.cat(logits_parts, dim=0)
    labels_all = torch.cat(label_parts, dim=0)
    valid_all = torch.cat(valid_parts, dim=0) > 0
    result: dict[str, Any] = {
        "active_views": view_mask.get_active_views(),
        "event_count": event_count,
        "valid_prediction_count": int(valid_all.sum().item()),
        "macro_f1": _macro_f1(
            logits_all,
            labels_all,
            valid_all,
            len(primary_task_spec["class_names"]),
        ),
        "accuracy": _accuracy(logits_all, labels_all, valid_all),
        "posterior_uncertainty": None,
    }
    if collect_uncertainty:
        uncertainty_all = torch.cat(uncertainty_parts, dim=0)
        predictions = logits_all.argmax(dim=-1)
        correct = predictions.eq(labels_all) & valid_all
        incorrect = predictions.ne(labels_all) & valid_all
        result["posterior_uncertainty"] = {
            "definition": "mean(exp(fused_posterior_logvar), latent_dimension)",
            "all": _uncertainty_group(uncertainty_all[valid_all]),
            "correct": _uncertainty_group(uncertainty_all[correct]),
            "incorrect": _uncertainty_group(uncertainty_all[incorrect]),
        }
    return result


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot compute a mean from an empty sequence.")
    return float(sum(materialized) / len(materialized))


def _pool_uncertainty(masked_tests: dict[str, Any], mask_names: Iterable[str], group: str) -> dict[str, Any]:
    total_count = 0
    weighted_sum = 0.0
    for mask_name in mask_names:
        stats = masked_tests[mask_name]["posterior_uncertainty"][group]
        count = int(stats["count"])
        mean_value = stats["mean"]
        if count and mean_value is not None:
            total_count += count
            weighted_sum += float(mean_value) * count
    return {"count": total_count, "mean": (weighted_sum / total_count if total_count else None)}


def _build_seed_metrics(fusion_variant: str, masked_tests: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any] | None]:
    robustness = {
        condition: _mean(masked_tests[name]["macro_f1"] for name in mask_names)
        for condition, mask_names in CONDITION_MASKS.items()
    }
    uncertainty = None
    if fusion_variant == "poe":
        uncertainty = {
            condition: {
                group: _pool_uncertainty(masked_tests, mask_names, group)
                for group in ("all", "correct", "incorrect")
            }
            for condition, mask_names in CONDITION_MASKS.items()
        }
    return robustness, uncertainty


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _prepare_run(
    args: argparse.Namespace, dataset_id: str, fusion_variant: str, seed: int
) -> tuple[argparse.Namespace, Any, dict[str, Any], dict[str, Any]]:
    run_args = copy.copy(args)
    config = resolve_fusion_config(dataset_id=dataset_id, mode="full")
    config.fusion_variant = fusion_variant
    config = apply_optimizer_config(config, normalize_optimizer_config(config.optimizer))
    config.active_views = ",".join(FULL_VIEWS)
    config.data_loader_seed = seed
    config.resolved_overrides = {
        **config.resolved_overrides,
        "active_views": config.active_views,
        "data_loader_seed": seed,
        "fusion_variant": fusion_variant,
    }
    run_args.batch_size = int(config.batch_size) if args.batch_size is None else int(args.batch_size)
    run_args.max_epoch = int(config.max_epoch) if args.max_epoch is None else int(args.max_epoch)
    run_args.candidate_epochs = (
        int(config.candidate_epochs) if args.candidate_epochs is None else int(args.candidate_epochs)
    )
    run_args.early_stop_patience = (
        int(config.early_stop_patience) if args.early_stop_patience is None else int(args.early_stop_patience)
    )
    if run_args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if run_args.max_epoch < 1:
        raise ValueError("max_epoch must be at least 1.")
    if run_args.candidate_epochs < 1:
        raise ValueError("candidate_epochs must be at least 1.")
    if run_args.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1.")
    config.max_epoch = run_args.max_epoch
    config.candidate_epochs = run_args.candidate_epochs
    for key, value in {
        "view_reconstruction_loss_weight": args.reconstruction_loss_weight,
        "primary_task_loss_weight": args.task_primary_loss_weight,
        "stance_auxiliary_loss_weight": args.aux_stance_loss_weight,
        "posterior_kl_loss_weight": args.kl_loss_weight,
    }.items():
        if value is not None:
            setattr(config, key, float(value))
            config.resolved_overrides = {**config.resolved_overrides, key: float(value)}
    optimizer_config = _resolve_optimizer_config(run_args, config)
    config = apply_optimizer_config(config, optimizer_config)
    apply_sample_latent_default(config, list(FULL_VIEWS))
    signature_payload = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "dataset_id": dataset_id,
        "fusion_variant": fusion_variant,
        "seed": seed,
        "fusion_config": asdict(config),
        "batch_size": run_args.batch_size,
        "max_epoch": run_args.max_epoch,
        "candidate_epochs": run_args.candidate_epochs,
        "max_steps": run_args.max_steps,
        "early_stop_patience": run_args.early_stop_patience,
        "early_stop_min_delta": run_args.early_stop_min_delta,
        "train_splits": parse_split_arg(run_args.splits),
        "eval_splits": parse_split_arg(run_args.eval_splits),
        "loader": {
            "num_workers": run_args.num_workers,
            "prefetch_factor": run_args.prefetch_factor,
            "disable_persistent_workers": run_args.disable_persistent_workers,
            "disable_pin_memory": run_args.disable_pin_memory,
        },
        "device": run_args.device,
        "amp": run_args.amp,
    }
    return run_args, config, optimizer_config, signature_payload


def _run_one(args: argparse.Namespace, dataset_id: str, fusion_variant: str, seed: int) -> dict[str, Any]:
    run_args, config, optimizer_config, signature_payload = _prepare_run(args, dataset_id, fusion_variant, seed)
    experiment_signature = _signature(signature_payload)
    run_root = Path(args.run_root)
    if not args.force:
        completed = _find_completed_run(run_root, fusion_variant, dataset_id, seed, experiment_signature)
        if completed is not None:
            print(f"Skipping completed run: dataset={dataset_id} variant={fusion_variant} seed={seed}")
            return completed

    started_at = _now_iso()
    started_perf = time.perf_counter()
    run_dir = (
        run_root
        / "runs"
        / fusion_variant
        / dataset_id
        / f"seed_{seed}"
        / _now_dirstamp()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = resolve_device(args.device)
    use_amp = resolve_amp(args.amp, device)
    save_selected_checkpoint = resolve_checkpoint_choice(args.checkpoint)
    train_splits = parse_split_arg(args.splits)
    eval_splits = parse_split_arg(args.eval_splits)
    full_mask = ViewMask(active_views=set(FULL_VIEWS))
    seed_everything(seed)

    write_json(
        run_dir / "run_config.json",
        {
            "status": "running",
            "experiment_signature": experiment_signature,
            "signature_payload": signature_payload,
            "run_dir": str(run_dir),
            "started_at": started_at,
        },
    )
    artifact_results = ensure_model_inputs(
        dataset_id=dataset_id,
        mode="full",
        active_views=list(FULL_VIEWS),
        config=config,
        data_root=data_root,
        input_root=data_root / "preprocessed",
        options=EnsureArtifactsOptions(),
    )
    print(f"Running dataset={dataset_id} variant={fusion_variant} seed={seed}")
    for result in artifact_results:
        print(f"  artifact {result.artifact_key}: {result.status} ({result.reason})")

    train_loader, train_count, selected_train_splits, feature_dims, train_loader_settings, task_spec, _ = (
        build_train_loader(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=run_args.batch_size,
            splits=train_splits,
            device=device,
            seed=seed,
            shuffle=True,
            num_workers_override=run_args.num_workers,
            prefetch_factor_override=run_args.prefetch_factor,
            persistent_workers_override=(False if run_args.disable_persistent_workers else None),
            pin_memory_override=(False if run_args.disable_pin_memory else None),
            required_views=set(FULL_VIEWS),
        )
    )
    eval_loader, eval_count, selected_eval_splits, eval_dims, eval_loader_settings, eval_task_spec = build_eval_loader(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=run_args.batch_size,
        splits=eval_splits,
        device=device,
        seed=seed,
        num_workers_override=run_args.num_workers,
        prefetch_factor_override=run_args.prefetch_factor,
        persistent_workers_override=(False if run_args.disable_persistent_workers else None),
        pin_memory_override=(False if run_args.disable_pin_memory else None),
        required_views=set(FULL_VIEWS),
    )
    if eval_dims != feature_dims:
        raise ValueError("Feature dimensions differ between training and validation loaders.")
    if eval_task_spec != task_spec:
        raise ValueError("Primary task spec differs between training and validation loaders.")

    model = create_fusion_model(
        config,
        feature_dims,
        primary_task_spec=task_spec,
        active_views=set(FULL_VIEWS),
    ).to(device)
    optimizer = build_hierarchical_optimizer(
        model,
        backbone_lr=float(optimizer_config["backbone"]["lr"]),
        backbone_weight_decay=float(optimizer_config["backbone"]["weight_decay"]),
        top_layer_lr=float(optimizer_config["top_layer"]["lr"]),
        top_layer_weight_decay=float(optimizer_config["top_layer"]["weight_decay"]),
        head_lr=float(optimizer_config["head"]["lr"]),
        head_weight_decay=float(optimizer_config["head"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **_scheduler_params(optimizer_config))
    candidate_start = max(1, run_args.max_epoch - run_args.candidate_epochs + 1)
    selected: dict[str, Any] = {"epoch": None, "monitor_value": None, "state_dict": None}

    def on_epoch_end(event: dict[str, Any]) -> None:
        record = event["epoch_record"]
        epoch = int(record.epoch)
        monitor = record.monitor_value
        if epoch < candidate_start or monitor is None:
            return
        current = selected["monitor_value"]
        if current is None or float(monitor) > float(current):
            selected["epoch"] = epoch
            selected["monitor_value"] = float(monitor)
            selected["state_dict"] = _clone_state_dict_to_cpu(event["model"])

    training_result = run_training(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        primary_task_spec=task_spec,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        max_epoch=run_args.max_epoch,
        max_steps=run_args.max_steps,
        early_stop_patience=run_args.early_stop_patience,
        early_stop_min_delta=run_args.early_stop_min_delta,
        checkpoint_dir=None,
        save_best_checkpoint=False,
        log_path=None,
        eval_log_path=None,
        epoch_log_path=run_dir / "epoch_log.jsonl",
        view_mask=full_mask,
        use_amp=use_amp,
        step_log_interval=run_args.log_every_steps,
        on_new_best=None,
        on_epoch_end=on_epoch_end,
        enable_early_stop=False,
    )
    if selected["state_dict"] is None:
        raise ValueError(
            f"No selectable epoch was produced; candidate range starts at epoch {candidate_start}. "
            "If max_steps is set, ensure training reaches that range."
        )
    model.load_state_dict(selected["state_dict"])
    del selected["state_dict"]

    checkpoint_path: Path | None = None
    if save_selected_checkpoint:
        checkpoint_path = run_dir / "checkpoint_selected.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            primary_task_spec=task_spec,
            step=int(training_result["steps_completed"]),
            epoch=int(selected["epoch"]),
            extra_state={
                "fusion_variant": fusion_variant,
                "active_views": list(FULL_VIEWS),
                "selection_monitor_value": selected["monitor_value"],
                "candidate_epoch_start": candidate_start,
            },
        )

    test_loader, test_count, selected_test_splits, test_dims, test_loader_settings, test_task_spec = (
        build_test_loader_with_fallback(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=run_args.batch_size,
            device=device,
            num_workers=run_args.num_workers,
            prefetch_factor=run_args.prefetch_factor,
            disable_persistent_workers=run_args.disable_persistent_workers,
            disable_pin_memory=run_args.disable_pin_memory,
            required_views=set(FULL_VIEWS),
        )
    )
    if test_dims != feature_dims:
        raise ValueError("Feature dimensions differ between training and test loaders.")
    if test_task_spec != task_spec:
        raise ValueError("Primary task spec differs between training and test loaders.")

    masked_tests: dict[str, Any] = {}
    for mask_name, active_views, display_name in MASK_SPECS:
        print(f"  testing mask={display_name} active_views={','.join(active_views)}")
        masked_tests[mask_name] = _run_masked_test(
            model=model,
            loader=test_loader,
            primary_task_spec=task_spec,
            device=device,
            use_amp=use_amp,
            view_mask=ViewMask(active_views=set(active_views)),
            collect_uncertainty=fusion_variant == "poe",
        )
        masked_tests[mask_name]["display_name"] = display_name

    robustness, uncertainty = _build_seed_metrics(fusion_variant, masked_tests)
    masked_rows: list[dict[str, Any]] = []
    for mask_name, _, display_name in MASK_SPECS:
        result = masked_tests[mask_name]
        row = {
            "dataset_id": dataset_id,
            "fusion_variant": fusion_variant,
            "seed": seed,
            "mask_name": mask_name,
            "display_name": display_name,
            "active_views": "|".join(result["active_views"]),
            "event_count": result["event_count"],
            "valid_prediction_count": result["valid_prediction_count"],
            "macro_f1": result["macro_f1"],
            "accuracy": result["accuracy"],
        }
        for group in ("all", "correct", "incorrect"):
            stats = (result["posterior_uncertainty"] or {}).get(group) or {}
            row[f"uncertainty_{group}_count"] = stats.get("count")
            row[f"uncertainty_{group}_mean"] = stats.get("mean")
        masked_rows.append(row)
    _write_csv(
        run_dir / "masked_test_results.csv",
        list(masked_rows[0]),
        masked_rows,
    )
    write_json(run_dir / "masked_test_results.json", {"masked_tests": masked_tests})

    ended_at = _now_iso()
    summary = {
        "status": "complete",
        "experiment_signature": experiment_signature,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "run": {
            "dataset_id": dataset_id,
            "fusion_variant": fusion_variant,
            "seed": seed,
            "run_dir": str(run_dir),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": max(0.0, time.perf_counter() - started_perf),
        },
        "selection": {
            "epoch": selected["epoch"],
            "monitor_value": selected["monitor_value"],
            "candidate_epoch_start": candidate_start,
            "candidate_epoch_end": run_args.max_epoch,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        "data": {
            "train_splits": selected_train_splits,
            "eval_splits": selected_eval_splits,
            "test_splits": selected_test_splits,
            "train_sample_count": train_count,
            "eval_sample_count": eval_count,
            "test_sample_count": test_count,
            "feature_dims": feature_dims,
            "primary_task_spec": task_spec,
            "train_loader_settings": train_loader_settings,
            "eval_loader_settings": eval_loader_settings,
            "test_loader_settings": test_loader_settings,
        },
        "signature_payload": signature_payload,
        "training": _jsonable(training_result),
        "masked_tests": masked_tests,
        "robustness": robustness,
        "posterior_uncertainty": uncertainty,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "run_config.json",
        {
            "status": "complete",
            "experiment_signature": experiment_signature,
            "signature_payload": signature_payload,
            "run_dir": str(run_dir),
            "started_at": started_at,
            "ended_at": ended_at,
        },
    )
    return summary


def _finite_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _mean_and_sample_std(values: Iterable[Any]) -> tuple[float | None, float | None, int]:
    numbers = _finite_values(values)
    if not numbers:
        return None, None, 0
    return statistics.fmean(numbers), (statistics.stdev(numbers) if len(numbers) >= 2 else None), len(numbers)


def _format_latex(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _robustness_latex(rows: list[dict[str, Any]], datasets: list[str], variants: list[str]) -> str:
    lookup = {(row["dataset_id"], row["fusion_variant"]): row for row in rows}
    columns = "l" + "ccc" * len(datasets)
    lines = [
        r"\begin{table*}[h]",
        r"    \centering",
        r"    \caption{Missing-view robustness. Models are not retrained after views are masked.}",
        r"    \label{tab:missing-view-robustness}",
        r"    \setlength{\tabcolsep}{5pt}",
        f"    \\begin{{tabular}}{{{columns}}}",
        r"    \hline",
        "    & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{dataset}}}" for dataset in datasets) + r" \\",
    ]
    if datasets:
        lines.append(f"    \\cline{{2-{1 + 3 * len(datasets)}}}")
    lines.append(
        "    Fusion strategy & "
        + " & ".join(item for _ in datasets for item in ("Complete", "One missing", "Two missing"))
        + r" \\"
    )
    lines.append(r"    \hline")
    for variant in variants:
        label = VARIANT_LABELS[variant]
        if variant == "poe":
            label = f"\\textbf{{{label}}}"
        values: list[str] = []
        for dataset in datasets:
            row = lookup.get((dataset, variant), {})
            values.extend(
                _format_latex(row.get(f"{condition}_mean"))
                for condition in ("complete", "one_missing", "two_missing")
            )
        lines.append(f"    {label} & " + " & ".join(values) + r" \\")
    lines.extend([r"    \hline", r"    \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _uncertainty_latex(rows: list[dict[str, Any]], datasets: list[str]) -> str:
    lookup = {(row["dataset_id"], row["condition"]): row for row in rows}
    lines = [
        r"\begin{table*}[h]",
        r"    \centering",
        r"    \caption{Covariance-based latent uncertainty under different view-availability conditions.}",
        r"    \label{tab:posterior-uncertainty}",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \begin{tabular}{llccc}",
        r"    \hline",
        r"    Dataset & Available-view condition & All & Correct & Incorrect \\",
        r"    \hline",
    ]
    for dataset in datasets:
        for condition in ("complete", "one_missing", "two_missing"):
            row = lookup.get((dataset, condition), {})
            values = [_format_latex(row.get(f"{group}_mean")) for group in ("all", "correct", "incorrect")]
            lines.append(
                f"    {dataset} & {CONDITION_LABELS[condition]} & " + " & ".join(values) + r" \\"
            )
    lines.extend([r"    \hline", r"    \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _write_aggregate_outputs(
    run_root: Path,
    summaries: list[dict[str, Any]],
    datasets: list[str],
    variants: list[str],
) -> None:
    robustness_seed_rows: list[dict[str, Any]] = []
    uncertainty_seed_rows: list[dict[str, Any]] = []
    for summary in summaries:
        run = summary["run"]
        robust_row = {
            "dataset_id": run["dataset_id"],
            "fusion_variant": run["fusion_variant"],
            "fusion_strategy": VARIANT_LABELS[run["fusion_variant"]],
            "seed": run["seed"],
            **summary["robustness"],
            "run_dir": run["run_dir"],
        }
        robustness_seed_rows.append(robust_row)
        if run["fusion_variant"] == "poe" and summary.get("posterior_uncertainty") is not None:
            for condition, groups in summary["posterior_uncertainty"].items():
                row = {
                    "dataset_id": run["dataset_id"],
                    "seed": run["seed"],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "run_dir": run["run_dir"],
                }
                for group in ("all", "correct", "incorrect"):
                    row[f"{group}_count"] = groups[group]["count"]
                    row[f"{group}_mean"] = groups[group]["mean"]
                uncertainty_seed_rows.append(row)
    robustness_seed_rows.sort(key=lambda row: (datasets.index(row["dataset_id"]), variants.index(row["fusion_variant"]), row["seed"]))
    uncertainty_seed_rows.sort(
        key=lambda row: (datasets.index(row["dataset_id"]), list(CONDITION_MASKS).index(row["condition"]), row["seed"])
    )

    robustness_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for variant in variants:
            selected = [
                row for row in robustness_seed_rows
                if row["dataset_id"] == dataset and row["fusion_variant"] == variant
            ]
            row: dict[str, Any] = {
                "dataset_id": dataset,
                "fusion_variant": variant,
                "fusion_strategy": VARIANT_LABELS[variant],
            }
            for condition in CONDITION_MASKS:
                mean_value, std_value, count = _mean_and_sample_std(item[condition] for item in selected)
                row[f"{condition}_mean"] = mean_value
                row[f"{condition}_std"] = std_value
                row[f"{condition}_seed_count"] = count
            robustness_rows.append(row)

    uncertainty_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for condition in CONDITION_MASKS:
            selected = [
                row for row in uncertainty_seed_rows
                if row["dataset_id"] == dataset and row["condition"] == condition
            ]
            row = {
                "dataset_id": dataset,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
            }
            for group in ("all", "correct", "incorrect"):
                values = _finite_values(item[f"{group}_mean"] for item in selected)
                row[f"{group}_mean"] = statistics.fmean(values) if values else None
                row[f"{group}_seed_count"] = len(values)
            uncertainty_rows.append(row)

    if robustness_seed_rows:
        _write_csv(run_root / "robustness_seed_results.csv", list(robustness_seed_rows[0]), robustness_seed_rows)
    if uncertainty_seed_rows:
        _write_csv(run_root / "posterior_uncertainty_seed_results.csv", list(uncertainty_seed_rows[0]), uncertainty_seed_rows)
    _write_csv(run_root / "robustness_aggregate.csv", list(robustness_rows[0]), robustness_rows)
    _write_csv(run_root / "posterior_uncertainty_aggregate.csv", list(uncertainty_rows[0]), uncertainty_rows)
    write_json(
        run_root / "robustness_results.json",
        {"seed_results": robustness_seed_rows, "aggregate": robustness_rows},
    )
    write_json(
        run_root / "posterior_uncertainty_results.json",
        {"seed_results": uncertainty_seed_rows, "aggregate": uncertainty_rows},
    )
    (run_root / "missing_view_robustness_table.tex").write_text(
        _robustness_latex(robustness_rows, datasets, variants), encoding="utf-8"
    )
    (run_root / "posterior_uncertainty_table.tex").write_text(
        _uncertainty_latex(uncertainty_rows, datasets), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    datasets, variants, seeds = _validate_grid(args)
    summaries: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for fusion_variant in variants:
            for seed in seeds:
                summaries.append(_run_one(args, dataset_id, fusion_variant, seed))
                _write_aggregate_outputs(Path(args.run_root), summaries, datasets, variants)
    print(f"Completed {len(summaries)} experiment configurations. Results: {Path(args.run_root)}")


if __name__ == "__main__":
    main()
