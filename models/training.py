from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from config.resolver import resolve_encoder_config, resolve_fusion_config
from models.fusion.hierarchical_optimizer import build_hierarchical_optimizer
from models.fusion.losses import build_loss_weights
from models.fusion.config import apply_optimizer_config, apply_sample_latent_default, normalize_optimizer_config
from models.fusion.model import UMCLFiveViewModel
from models.fusion.training import build_eval_loader, build_train_loader, run_training
from models.fusion.view_mask import create_view_mask_from_config, get_mask_name
from models.artifacts import EnsureArtifactsOptions, ensure_model_inputs
from models.evaluation import build_test_loader_with_fallback, run_test_from_model
from utils.jsonl import write_json
from utils.torch_runtime import seed_everything


def run_training_cli(args: argparse.Namespace) -> None:
    from train_umcl import (
        _append_result_umcl_row,
        _build_core_hyperparameters,
        _build_core_summary_hyperparameters,
        _build_eval_log_row,
        _build_run_config,
        _build_summary,
        _eval_log_fieldnames,
        _now_dirstamp,
        _now_iso,
        _resolve_optimizer_config,
        _scheduler_params,
        _select_final_best_epoch,
        parse_split_arg,
        resolve_amp,
        resolve_checkpoint_choice,
        resolve_device,
    )

    dataset_id = args.dataset_id.strip()
    train_splits = parse_split_arg(args.splits)
    eval_splits = parse_split_arg(args.eval_splits)
    mode = "full"
    data_root = Path(args.data_root)
    device = resolve_device(args.device)
    use_amp = resolve_amp(args.amp, device)
    save_best_checkpoint = resolve_checkpoint_choice(args.checkpoint)
    seed = int(args.seed)
    run_started_at = _now_iso()
    run_started_perf = time.perf_counter()
    config = resolve_fusion_config(dataset_id=dataset_id, mode=mode)
    config = apply_optimizer_config(config, normalize_optimizer_config(config.optimizer))
    config.data_loader_seed = seed
    config.resolved_overrides = {
        **config.resolved_overrides,
        "data_loader_seed": config.data_loader_seed,
    }
    if args.batch_size is None:
        args.batch_size = int(config.batch_size)
    if args.early_stop_patience is None:
        args.early_stop_patience = int(config.early_stop_patience)
    if args.max_epoch is None:
        args.max_epoch = int(config.max_epoch)
    if args.max_epoch < 1:
        raise ValueError("max_epoch must be at least 1.")
    config.max_epoch = int(args.max_epoch)
    config.resolved_overrides = {
        **config.resolved_overrides,
        "max_epoch": config.max_epoch,
    }
    if args.candidate_epochs is None:
        args.candidate_epochs = int(config.candidate_epochs)
    if args.candidate_epochs < 1:
        raise ValueError("candidate_epochs must be at least 1.")
    config.candidate_epochs = int(args.candidate_epochs)
    config.resolved_overrides = {
        **config.resolved_overrides,
        "candidate_epochs": config.candidate_epochs,
    }
    loss_weight_overrides = {
        "view_reconstruction_loss_weight": args.reconstruction_loss_weight,
        "primary_task_loss_weight": args.task_primary_loss_weight,
        "stance_auxiliary_loss_weight": args.aux_stance_loss_weight,
        "posterior_kl_loss_weight": args.kl_loss_weight,
    }
    for key, value in loss_weight_overrides.items():
        if value is None:
            continue
        setattr(config, key, float(value))
        config.resolved_overrides = {
            **config.resolved_overrides,
            key: getattr(config, key),
        }
    optimizer_config = _resolve_optimizer_config(args, config)
    config = apply_optimizer_config(config, optimizer_config)
    seed_everything(seed)
    active_views_config = args.active_views if args.active_views is not None else config.active_views
    view_mask = create_view_mask_from_config(active_views_config)
    active_views = view_mask.get_active_views()
    apply_sample_latent_default(config, active_views)
    artifact_results = ensure_model_inputs(
        dataset_id=dataset_id,
        mode=mode,
        active_views=active_views,
        config=config,
        data_root=data_root,
        input_root=data_root / "preprocessed",
        options=EnsureArtifactsOptions(),
    )

    run_time_at = _now_dirstamp()
    seed_dir_name = f"seed_{seed}"
    run_dir = Path(args.run_root) / dataset_id / seed_dir_name / run_time_at
    run_dir.mkdir(parents=True, exist_ok=True)
    core_hyperparameters = _build_core_hyperparameters(
        args=args,
        optimizer_config=optimizer_config,
        config=config,
        data_root=data_root,
        device=device,
        use_amp=use_amp,
        train_splits=train_splits,
        eval_splits=eval_splits,
        active_views=active_views,
        save_best_checkpoint=save_best_checkpoint,
    )
    loss_weights = build_loss_weights(config)
    print("Artifact ensure results:")
    for result in artifact_results:
        print(f"  {result.artifact_key}: {result.status} ({result.reason})")

    train_loader, train_sample_count, selected_train_splits, feature_dims, _train_loader_settings, primary_task_spec, _primary_label_counts = build_train_loader(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=args.batch_size,
        splits=train_splits,
        device=device,
        seed=seed,
        shuffle=True,
        num_workers_override=args.num_workers,
        prefetch_factor_override=args.prefetch_factor,
        persistent_workers_override=(False if args.disable_persistent_workers else None),
        pin_memory_override=(False if args.disable_pin_memory else None),
        required_views=set(active_views),
    )
    eval_loader, eval_sample_count, selected_eval_splits, eval_feature_dims, _eval_loader_settings, eval_primary_task_spec = build_eval_loader(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=args.batch_size,
        splits=eval_splits,
        device=device,
        seed=seed,
        num_workers_override=args.num_workers,
        prefetch_factor_override=args.prefetch_factor,
        persistent_workers_override=(False if args.disable_persistent_workers else None),
        pin_memory_override=(False if args.disable_pin_memory else None),
        required_views=set(active_views),
    )
    if feature_dims != eval_feature_dims:
        raise ValueError("Feature dimensions differ between training and evaluation loaders.")
    if primary_task_spec != eval_primary_task_spec:
        raise ValueError("Primary task spec differs between training and evaluation loaders.")

    core_hyperparameters["train_splits"] = selected_train_splits
    core_hyperparameters["eval_splits"] = selected_eval_splits

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **_scheduler_params(optimizer_config))

    write_json(
        run_dir / "run_config.json",
        _build_run_config(
            dataset_id=dataset_id,
            mode=mode,
            seed=seed,
            run_dir=run_dir,
            start_time=run_started_at,
            end_time=None,
            core_hyperparameters=core_hyperparameters,
        ),
    )

    mask_name = get_mask_name(view_mask)
    print(f"Using view mask: {mask_name}")
    print(f"  Active views: {sorted(active_views)}")

    stance_config = resolve_encoder_config("stance", dataset_id, mode)
    propagation_config = resolve_encoder_config("propagation", dataset_id, mode)
    latest_test_summary: dict[str, Any] | None = None
    latest_test_meta: dict[str, Any] | None = None
    test_summaries_by_epoch: dict[int, dict[str, Any]] = {}
    eval_rows_by_epoch: dict[int, dict[str, Any]] = {}
    checkpoint_meta_by_epoch: dict[int, dict[str, Any]] = {}
    test_start_epoch = max(1, int(args.max_epoch) - int(args.candidate_epochs) + 1)
    eval_log_path = run_dir / "eval_test_report.csv"
    eval_log_fieldnames = _eval_log_fieldnames(list(primary_task_spec["class_names"]))
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("w", encoding="utf-8", newline="") as eval_log_handle:
        eval_log_writer = csv.DictWriter(eval_log_handle, fieldnames=eval_log_fieldnames)
        eval_log_writer.writeheader()

        def on_new_best(event: dict[str, Any]) -> None:
            epoch_record = event["epoch_record"]
            checkpoint_path = event["checkpoint_path"]
            checkpoint_saved = bool(event["checkpoint_saved"])
            best_epoch = int(epoch_record.best_epoch if epoch_record.best_epoch is not None else event["epoch"])
            best_monitor_value = float(
                epoch_record.best_monitor_value if epoch_record.best_monitor_value is not None else event["monitor_value"]
            )
            checkpoint_meta_by_epoch[best_epoch] = {
                "best_epoch": best_epoch,
                "best_monitor_value": best_monitor_value,
                "checkpoint_saved": checkpoint_saved,
                "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else None),
            }

        def on_epoch_end(event: dict[str, Any]) -> None:
            nonlocal latest_test_summary
            epoch_record = event["epoch_record"]
            epoch = int(epoch_record.epoch)
            if epoch >= test_start_epoch:
                checkpoint_meta = checkpoint_meta_by_epoch.get(epoch, {})
                checkpoint_path_value = checkpoint_meta.get("checkpoint_path")
                test_loader, sample_count, selected_splits, test_feature_dims, loader_settings, test_primary_task_spec = build_test_loader_with_fallback(
                    dataset_id=dataset_id,
                    config=config,
                    data_root=data_root,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    disable_persistent_workers=args.disable_persistent_workers,
                    disable_pin_memory=args.disable_pin_memory,
                    required_views=set(active_views),
                )
                if test_feature_dims != feature_dims:
                    raise ValueError("Feature dimensions differ between training and test loaders.")
                if test_primary_task_spec != primary_task_spec:
                    raise ValueError("Primary task spec differs between training and test loaders.")

                latest_test_summary = run_test_from_model(
                    model=event["model"],
                    dataset_id=dataset_id,
                    mode=mode,
                    config=config,
                    primary_task_spec=primary_task_spec,
                    loader=test_loader,
                    sample_count=sample_count,
                    selected_splits=selected_splits,
                    feature_dims=test_feature_dims,
                    batch_size=args.batch_size,
                    device=device,
                    use_amp=use_amp,
                    active_views=active_views,
                    checkpoint_path=(Path(checkpoint_path_value) if checkpoint_path_value is not None else None),
                    max_batches=None,
                    output_path=None,
                    log_path=None,
                    loader_settings=loader_settings,
                    loader_runtime_overrides={
                        "num_workers": args.num_workers,
                        "prefetch_factor": args.prefetch_factor,
                        "disable_persistent_workers": args.disable_persistent_workers,
                        "disable_pin_memory": args.disable_pin_memory,
                    },
                    stance_config=stance_config,
                    propagation_config=propagation_config,
                )
                test_summaries_by_epoch[epoch] = latest_test_summary
            row = _build_eval_log_row(
                record=epoch_record,
                dataset_id=dataset_id,
                seed=seed,
                view_reconstruction_loss_weight=float(loss_weights["view_reconstruction_loss_weight"]),
                posterior_kl_loss_weight=float(loss_weights["posterior_kl_loss_weight"]),
                active_views=active_views,
                class_names=list(primary_task_spec["class_names"]),
                test_summary=test_summaries_by_epoch.get(epoch),
            )
            eval_rows_by_epoch[epoch] = row
            eval_log_writer.writerow(row)
            eval_log_handle.flush()

        training_result = run_training(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            primary_task_spec=primary_task_spec,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            max_epoch=args.max_epoch,
            max_steps=args.max_steps,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            checkpoint_dir=run_dir,
            log_path=None,
            eval_log_path=None,
            epoch_log_path=run_dir / "epoch_log.jsonl",
            view_mask=view_mask,
            use_amp=use_amp,
            step_log_interval=args.log_every_steps,
            save_best_checkpoint=save_best_checkpoint,
            on_new_best=on_new_best,
            on_epoch_end=on_epoch_end,
            enable_early_stop=False,
        )
    last_step = training_result["last_step_record"]
    epoch_records = training_result["epoch_records"]
    if last_step is None or not epoch_records:
        raise ValueError("Training did not produce any records.")
    last_epoch = epoch_records[-1]
    run_ended_at = _now_iso()
    run_duration_seconds = max(0.0, time.perf_counter() - run_started_perf)
    final_best_record, final_best_candidates = _select_final_best_epoch(epoch_records, test_start_epoch=test_start_epoch)
    final_best_row: dict[str, Any] | None = None
    if final_best_record is not None:
        final_best_epoch = int(final_best_record.epoch)
        final_best_row = eval_rows_by_epoch.get(final_best_epoch)
        latest_test_summary = test_summaries_by_epoch.get(final_best_epoch)
        checkpoint_meta = checkpoint_meta_by_epoch.get(final_best_epoch, {})
        latest_test_meta = {
            "best_epoch": final_best_epoch,
            "best_monitor_value": final_best_record.monitor_value,
            "best_train_task_primary_loss": final_best_record.train_losses.get("task_primary_loss"),
            "checkpoint_saved": checkpoint_meta.get("checkpoint_saved", False),
            "checkpoint_path": checkpoint_meta.get("checkpoint_path"),
            "candidate_epoch_start": test_start_epoch,
            "candidate_epoch_end": int(args.max_epoch),
            "candidate_epochs": int(args.candidate_epochs),
            "selection_candidates": final_best_candidates,
        }
        if final_best_row is not None:
            _append_result_umcl_row(
                Path(__file__).resolve().parents[1] / "result_umcl.csv",
                eval_log_fieldnames,
                final_best_row,
                timestamp=run_ended_at,
            )
    write_json(
        run_dir / "run_config.json",
        _build_run_config(
            dataset_id=dataset_id,
            mode=mode,
            seed=seed,
            run_dir=run_dir,
            start_time=run_started_at,
            end_time=run_ended_at,
            core_hyperparameters=core_hyperparameters,
        ),
    )

    summary = _build_summary(
        dataset_id=dataset_id,
        mode=mode,
        seed=seed,
        run_dir=run_dir,
        start_time=run_started_at,
        end_time=run_ended_at,
        duration_seconds=run_duration_seconds,
        active_views=active_views,
        train_splits=selected_train_splits,
        eval_splits=selected_eval_splits,
        train_sample_count=train_sample_count,
        eval_sample_count=eval_sample_count,
        feature_dims=feature_dims,
        primary_task_spec=primary_task_spec,
        core_hyperparameters=_build_core_summary_hyperparameters(
            args=args,
            optimizer_config=optimizer_config,
            config=config,
            device=device,
            use_amp=use_amp,
            active_views=active_views,
            train_splits=selected_train_splits,
            eval_splits=selected_eval_splits,
        ),
        training_result=training_result,
        last_step=last_step,
        last_epoch=last_epoch,
    )
    summary["testing"] = {
        "auto_test_ran": bool(latest_test_summary is not None),
        "best_epoch": (latest_test_meta["best_epoch"] if latest_test_meta is not None else None),
        "best_monitor_value": (latest_test_meta["best_monitor_value"] if latest_test_meta is not None else None),
        "best_train_task_primary_loss": (
            latest_test_meta.get("best_train_task_primary_loss") if latest_test_meta is not None else None
        ),
        "candidate_epoch_start": (latest_test_meta.get("candidate_epoch_start") if latest_test_meta is not None else test_start_epoch),
        "candidate_epoch_end": (latest_test_meta.get("candidate_epoch_end") if latest_test_meta is not None else int(args.max_epoch)),
        "candidate_epochs": int(args.candidate_epochs),
        "selection_candidates": (latest_test_meta.get("selection_candidates") if latest_test_meta is not None else []),
        "checkpoint_saved": (latest_test_meta["checkpoint_saved"] if latest_test_meta is not None else None),
        "checkpoint_path": (latest_test_meta["checkpoint_path"] if latest_test_meta is not None else None),
        "macro_f1": (latest_test_summary["macro_f1"] if latest_test_summary is not None else None),
        "accuracy": (latest_test_summary["accuracy"] if latest_test_summary is not None else None),
        "per_class_f1": (latest_test_summary["per_class_f1"] if latest_test_summary is not None else None),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary["testing"], ensure_ascii=False, indent=2))
