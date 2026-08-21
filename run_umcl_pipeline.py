from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from config.resolver import (
    reset_pretrained_model_override,
    resolve_encoder_config,
    resolve_fusion_config,
    set_pretrained_model_override,
)
from models.artifacts import EnsureArtifactsOptions, ensure_model_inputs
from models.evaluation import run_test
from models.fusion.config import apply_optimizer_config, apply_sample_latent_default, normalize_optimizer_config
from models.fusion.hierarchical_optimizer import build_hierarchical_optimizer
from models.fusion.model import UMCLFiveViewModel
from models.fusion.training import build_eval_loader, build_train_loader, run_training
from models.fusion.view_mask import create_view_mask_from_config
from utils.jsonl import dataclass_to_dict, write_json
from utils.torch_runtime import seed_everything


STRICT_TRAIN_SPLITS = ["train"]
STRICT_VALIDATION_SPLITS = ["val"]
STRICT_TEST_SPLITS = ["test"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standard UMCL train-validation-test pipeline.",
    )
    parser.add_argument("--dataset_id", required=True, help="Single dataset id, e.g. RumourEval.")
    parser.add_argument(
        "--pretrained_model",
        default=None,
        help=(
            "HuggingFace model id or local model path used by the content and stance encoders. "
            "Default: use config/config.json (currently xlm-roberta-base)."
        ),
    )
    parser.add_argument("--seed", type=int, default=2, help="Global random seed. Default: 2.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_root", default="runs")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _now_dirstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _resolved_optimizer_config(config) -> dict[str, Any]:
    optimizer_config = normalize_optimizer_config(config.optimizer)
    missing: list[str] = []
    for group_name in ("backbone", "top_layer", "head"):
        for field_name in ("lr", "weight_decay"):
            if optimizer_config[group_name][field_name] is None:
                missing.append(f"optimizer.{group_name}.{field_name}")
    for field_name in ("factor", "patience", "min_lr", "start_epoch"):
        if optimizer_config["scheduler"][field_name] is None:
            missing.append(f"optimizer.scheduler.{field_name}")
    if missing:
        raise ValueError(f"Missing standard pipeline config values: {missing}")
    return optimizer_config


def _scheduler_params(optimizer_config: dict[str, Any]) -> dict[str, Any]:
    scheduler_config = optimizer_config["scheduler"]
    return {
        "mode": "max",
        "factor": float(scheduler_config["factor"]),
        "patience": int(scheduler_config["patience"]),
        "min_lr": float(scheduler_config["min_lr"]),
    }


def _run_config(
    *,
    dataset_id: str,
    seed: int,
    run_dir: Path,
    start_time: str,
    end_time: str | None,
    device: torch.device,
    use_amp: bool,
    pretrained_models: dict[str, str],
    active_views: list[str],
    config,
) -> dict[str, Any]:
    return {
        "project": "UMCL",
        "entrypoint": "run_umcl_pipeline.py",
        "run": {
            "dataset_id": dataset_id,
            "seed": seed,
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": end_time,
        },
        "protocol": {
            "train_splits": STRICT_TRAIN_SPLITS,
            "validation_splits": STRICT_VALIDATION_SPLITS,
            "test_splits": STRICT_TEST_SPLITS,
            "selection": "best validation monitor",
            "test_evaluations": 1,
        },
        "runtime": {
            "device": str(device),
            "use_amp": use_amp,
            "pretrained_models": pretrained_models,
            "active_views": active_views,
        },
        "fusion_config": config.as_dict(),
    }


def _build_summary(
    *,
    run_config: dict[str, Any],
    duration_seconds: float,
    artifact_results: list[Any],
    train_sample_count: int,
    validation_sample_count: int,
    feature_dims: dict[str, int],
    primary_task_spec: dict[str, Any],
    training_result: dict[str, Any],
    test_summary: dict[str, Any],
) -> dict[str, Any]:
    epoch_records = list(training_result["epoch_records"])
    return {
        "run": {
            **run_config["run"],
            "duration_seconds": duration_seconds,
        },
        "protocol": run_config["protocol"],
        "runtime": run_config["runtime"],
        "data": {
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "test_sample_count": test_summary.get("sample_count"),
            "feature_dims": feature_dims,
            "primary_task_spec": primary_task_spec,
        },
        "artifacts": [dataclass_to_dict(result) for result in artifact_results],
        "training": {
            "epochs_completed": training_result["epochs_completed"],
            "steps_completed": training_result["steps_completed"],
            "stopped_early": training_result["stopped_early"],
            "stop_reason": training_result["stop_reason"],
            "monitor_name": training_result["monitor_name"],
            "best_epoch": training_result["best_epoch"],
            "best_monitor_value": training_result["best_monitor_value"],
            "best_checkpoint": training_result["best_checkpoint"],
            "final_epoch_record": dataclass_to_dict(epoch_records[-1]) if epoch_records else None,
            "cache_stats": training_result.get("cache_stats"),
        },
        "testing": test_summary,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    override_token = set_pretrained_model_override(args.pretrained_model)
    try:
        return _run_pipeline(args)
    finally:
        reset_pretrained_model_override(override_token)


def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    dataset_id = str(args.dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty.")

    mode = "full"
    seed = int(args.seed)
    data_root = Path(args.data_root)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    started_at = _now_iso()
    started_perf = time.perf_counter()

    config = resolve_fusion_config(dataset_id=dataset_id, mode=mode)
    optimizer_config = _resolved_optimizer_config(config)
    config = apply_optimizer_config(config, optimizer_config)
    config.data_loader_seed = seed
    config.resolved_overrides = {**config.resolved_overrides, "data_loader_seed": seed}

    view_mask = create_view_mask_from_config(config.active_views)
    active_views = view_mask.get_active_views()
    apply_sample_latent_default(config, active_views)
    pretrained_models = {
        view_name: resolve_encoder_config(view_name, dataset_id, mode).model_name(mode)
        for view_name in ("content", "stance")
    }
    seed_everything(seed)

    artifact_results = ensure_model_inputs(
        dataset_id=dataset_id,
        mode=mode,
        active_views=active_views,
        config=config,
        data_root=data_root,
        input_root=data_root / "preprocessed",
        options=EnsureArtifactsOptions(),
    )

    run_dir = Path(args.run_root) / dataset_id / f"seed_{seed}" / _now_dirstamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = _run_config(
        dataset_id=dataset_id,
        seed=seed,
        run_dir=run_dir,
        start_time=started_at,
        end_time=None,
        device=device,
        use_amp=use_amp,
        pretrained_models=pretrained_models,
        active_views=active_views,
        config=config,
    )
    write_json(run_dir / "run_config.json", run_config)

    train_loader, train_count, train_splits, feature_dims, _, primary_task_spec, _ = build_train_loader(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=int(config.batch_size),
        splits=STRICT_TRAIN_SPLITS,
        device=device,
        seed=seed,
        shuffle=True,
        required_views=set(active_views),
    )
    validation_loader, validation_count, validation_splits, validation_dims, _, validation_task_spec = build_eval_loader(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=int(config.batch_size),
        splits=STRICT_VALIDATION_SPLITS,
        device=device,
        seed=seed,
        required_views=set(active_views),
    )
    if train_splits != STRICT_TRAIN_SPLITS or validation_splits != STRICT_VALIDATION_SPLITS:
        raise ValueError("The standard pipeline requires exact train and val splits.")
    if feature_dims != validation_dims:
        raise ValueError("Feature dimensions differ between train and validation splits.")
    if primary_task_spec != validation_task_spec:
        raise ValueError("Primary task spec differs between train and validation splits.")

    model = UMCLFiveViewModel(
        config,
        feature_dims,
        primary_task_spec=primary_task_spec,
        active_views=set(active_views),
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **_scheduler_params(optimizer_config),
    )

    training_result = run_training(
        model=model,
        train_loader=train_loader,
        eval_loader=validation_loader,
        primary_task_spec=primary_task_spec,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        max_epoch=int(config.max_epoch),
        early_stop_patience=int(config.early_stop_patience),
        checkpoint_dir=run_dir,
        save_best_checkpoint=True,
        eval_log_path=run_dir / "validation_log.csv",
        epoch_log_path=run_dir / "epoch_log.jsonl",
        view_mask=view_mask,
        use_amp=use_amp,
        enable_early_stop=True,
    )

    checkpoint_value = training_result.get("best_checkpoint")
    if not checkpoint_value:
        raise RuntimeError("Training did not report a best checkpoint.")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {checkpoint_path}")

    test_summary = run_test(
        dataset_id=dataset_id,
        checkpoint_path=checkpoint_path,
        mode=mode,
        data_root=data_root,
        batch_size=int(config.batch_size),
        num_workers=None,
        prefetch_factor=None,
        disable_persistent_workers=False,
        disable_pin_memory=False,
        max_batches=None,
        splits=STRICT_TEST_SPLITS,
        device=device,
        use_amp=use_amp,
        output_path=run_dir / "test_summary.json",
        log_path=None,
    )

    ended_at = _now_iso()
    duration_seconds = max(0.0, time.perf_counter() - started_perf)
    completed_run_config = _run_config(
        dataset_id=dataset_id,
        seed=seed,
        run_dir=run_dir,
        start_time=started_at,
        end_time=ended_at,
        device=device,
        use_amp=use_amp,
        pretrained_models=pretrained_models,
        active_views=active_views,
        config=config,
    )
    write_json(run_dir / "run_config.json", completed_run_config)
    summary = _build_summary(
        run_config=completed_run_config,
        duration_seconds=duration_seconds,
        artifact_results=artifact_results,
        train_sample_count=train_count,
        validation_sample_count=validation_count,
        feature_dims=feature_dims,
        primary_task_spec=primary_task_spec,
        training_result=training_result,
        test_summary=test_summary,
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_pipeline(parse_args())
    print(json.dumps({"run": summary["run"], "training": summary["training"], "testing": summary["testing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
