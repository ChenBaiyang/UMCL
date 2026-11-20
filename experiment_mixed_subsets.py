"""UMCL experiment: mixed subsets."""

from __future__ import annotations

from utils.runtime import (
    resolve_device as _resolve_device, resolve_amp as _resolve_amp,
    resolve_checkpoint_choice as _resolve_checkpoint_choice,
)

import argparse
import copy
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from config.resolver import (
    reset_pretrained_model_override,
    resolve_fusion_config,
    set_pretrained_model_override,
)
from models.artifacts import EnsureArtifactsOptions, ensure_model_inputs
from models.evaluation import build_test_loader_with_fallback
from models.fusion.config import (
    apply_optimizer_config,
    apply_sample_latent_default,
    normalize_optimizer_config,
)
from models.fusion.factory import create_fusion_model
from models.fusion.hierarchical_optimizer import build_hierarchical_optimizer
from models.fusion.subset_sampling import (
    CORE_VIEWS,
    SUBSET_SPECS,
    DeterministicMixedSubsetMasker,
)
from models.fusion.training import (
    BEST_SELECTION_RULE_RAW_MAX,
    BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
    build_eval_loader,
    build_train_loader,
    run_training,
    save_checkpoint,
)
from models.fusion.view_mask import ViewMask
from experiment_missing_views import (
    _clone_state_dict_to_cpu,
    _jsonable,
    _mean_and_sample_std,
    _run_masked_test,
    _signature,
    _write_csv,
)
from train_umcl import (
    _now_dirstamp,
    _now_iso,
    _resolve_optimizer_config,
    _scheduler_params,
    parse_split_arg,
)
from utils.jsonl import write_json
from utils.torch_runtime import seed_everything


DEFAULT_DATASETS = ("PHEME", "Weibo")
DEFAULT_SEEDS = (2, 4, 8, 16, 32)
DEFAULT_REGIMES = ("complete_only", "mixed")
EXPERIMENT_VERSION = 3
STRICT_TEST_SPLITS = ["test"]
REFERENCE_WARNING_TOLERANCE = 1e-4
REGIME_LABELS = {
    "subset_specific": "Subset-specific models",
    "complete_only": "Complete-only UMCL",
    "mixed": "Unified mixed-subset UMCL",
}
SUBSET_ORDER = [name for name, _ in SUBSET_SPECS]
SUBSET_TO_VIEWS = {name: views for name, views in SUBSET_SPECS}
ACTIVE_VIEWS_TO_SUBSET = {
    "|".join(views): name for name, views in SUBSET_SPECS
}


def _comma_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _seed_list(raw: str) -> list[int]:
    values = [int(item) for item in _comma_list(raw)]
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values








def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train event-level mixed-subset UMCL and evaluate standard C+S+P checkpoints "
            "under seven observed-view masks."
        )
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    parser.add_argument("--seeds", type=_seed_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_root", default="runs_unified_mixed_subset")
    parser.add_argument("--specialist_results", default="results/result_view_contribution.csv")
    parser.add_argument(
        "--complete_only_checkpoint_manifest",
        default="results/standard_umcl_checkpoints.csv",
        help=(
            "CSV with dataset_id,seed,checkpoint_path and optional pretrained_model columns. "
            "Relative checkpoint paths are resolved from the manifest directory."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require_complete_grid", action="store_true")
    parser.add_argument("--kl_loss_weight", type=float, default=None)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=None)
    parser.add_argument("--task_primary_loss_weight", type=float, default=None)
    parser.add_argument("--aux_stance_loss_weight", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--candidate_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        default="false",
        choices=["true", "false"],
        help="Save the selected mixed-subset checkpoint; ignored by complete_only.",
    )
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
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    return parser.parse_args()


def _validate_grid(args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    datasets = _comma_list(args.datasets)
    regimes = [item.lower() for item in _comma_list(args.regimes)]
    seeds = list(args.seeds)
    invalid_datasets = sorted(set(datasets) - set(DEFAULT_DATASETS))
    invalid_regimes = sorted(set(regimes) - set(DEFAULT_REGIMES))
    if invalid_datasets:
        raise ValueError(f"This experiment supports only {list(DEFAULT_DATASETS)}; got {invalid_datasets}.")
    if invalid_regimes:
        raise ValueError(f"Unsupported regimes {invalid_regimes}; expected {list(DEFAULT_REGIMES)}.")
    if not datasets or not regimes or not seeds:
        raise ValueError("datasets, regimes, and seeds must all be non-empty.")
    for values, label in ((datasets, "datasets"), (regimes, "regimes"), (seeds, "seeds")):
        if len(values) != len(set(values)):
            raise ValueError(f"{label} must not contain duplicates.")
    return datasets, regimes, seeds


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_active_views(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = []
    return tuple(dict.fromkeys(values))


def _normalize_checkpoint_active_views(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("active_views")
    if raw is None:
        raw = (payload.get("fusion_config") or {}).get("active_views")
    return _normalize_active_views(raw)


def _infer_pretrained_model(checkpoint_path: Path, explicit_value: Any) -> tuple[str | None, str]:
    if explicit_value is not None and not pd.isna(explicit_value):
        value = str(explicit_value).strip()
        if value:
            return value, "manifest"
    run_config_path = checkpoint_path.parent / "run_config.json"
    if run_config_path.is_file():
        run_config = _read_json(run_config_path)
        pretrained = ((run_config.get("runtime") or {}).get("pretrained_models") or {})
        values = {
            str(pretrained.get(view)).strip()
            for view in ("content", "stance")
            if pretrained.get(view) is not None and str(pretrained.get(view)).strip()
        }
        if len(values) == 1:
            return next(iter(values)), "sibling_run_config"
        if len(values) > 1:
            raise ValueError(
                f"Checkpoint run_config uses different content/stance pretrained models, which the "
                f"shared override cannot reproduce: {run_config_path}"
            )
    return None, "configured_default"


def _validate_checkpoint_payload(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    dataset_id: str,
    seed: int,
) -> dict[str, Any]:
    required = {"model_state_dict", "fusion_config", "primary_task_spec"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} lacks required keys: {sorted(missing)}")
    fusion_config = payload["fusion_config"]
    if not isinstance(fusion_config, dict):
        raise TypeError(f"Checkpoint fusion_config must be a dict: {checkpoint_path}")
    checkpoint_dataset = fusion_config.get("dataset_id")
    if checkpoint_dataset != dataset_id:
        raise ValueError(
            f"Checkpoint dataset mismatch for {checkpoint_path}: expected {dataset_id}, got {checkpoint_dataset}"
        )
    checkpoint_seed = payload.get("seed", fusion_config.get("data_loader_seed"))
    if checkpoint_seed is None or int(checkpoint_seed) != int(seed):
        raise ValueError(
            f"Checkpoint seed mismatch for {checkpoint_path}: expected {seed}, got {checkpoint_seed}"
        )
    if payload.get("seed") is not None and fusion_config.get("data_loader_seed") is not None:
        if int(payload["seed"]) != int(fusion_config["data_loader_seed"]):
            raise ValueError(
                f"Checkpoint contains conflicting seed identities in payload and fusion_config: "
                f"{checkpoint_path}"
            )
    checkpoint_mode = str(fusion_config.get("mode", "full")).strip().lower()
    if checkpoint_mode != "full":
        raise ValueError(f"Complete-only checkpoint must use full mode, got {checkpoint_mode!r}.")
    fusion_variant = str(fusion_config.get("fusion_variant", "poe")).strip().lower()
    if fusion_variant != "poe":
        raise ValueError(
            f"Complete-only checkpoint must use PoE fusion, got {fusion_variant!r}: {checkpoint_path}"
        )
    active_views = _normalize_checkpoint_active_views(payload)
    if len(active_views) != len(CORE_VIEWS) or set(active_views) != set(CORE_VIEWS):
        raise ValueError(
            f"Complete-only checkpoint must contain exactly C/S/P; got {list(active_views)} in {checkpoint_path}"
        )
    configured_views = _normalize_active_views(fusion_config.get("active_views"))
    if len(configured_views) != len(CORE_VIEWS) or set(configured_views) != set(CORE_VIEWS):
        raise ValueError(
            f"Checkpoint fusion_config must contain exactly C/S/P; got {list(configured_views)} "
            f"in {checkpoint_path}"
        )
    if not isinstance(payload["model_state_dict"], dict) or not payload["model_state_dict"]:
        raise ValueError(f"Checkpoint model_state_dict is empty: {checkpoint_path}")
    if not isinstance(payload["primary_task_spec"], dict):
        raise TypeError(f"Checkpoint primary_task_spec must be a dict: {checkpoint_path}")
    selection_scope = str(payload.get("selection_scope") or "").strip()
    selection_rule = str(payload.get("selection_rule") or "").strip()
    if selection_scope != BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS:
        raise ValueError(
            f"Checkpoint is not candidate-epoch selected: expected selection_scope="
            f"{BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS!r}, got {selection_scope!r}: {checkpoint_path}"
        )
    if selection_rule != BEST_SELECTION_RULE_RAW_MAX:
        raise ValueError(
            f"Checkpoint uses an incompatible selection rule: expected "
            f"{BEST_SELECTION_RULE_RAW_MAX!r}, got {selection_rule!r}: {checkpoint_path}"
        )
    try:
        checkpoint_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        candidate_start = int(payload["candidate_epoch_start"])
        candidate_end = int(payload["candidate_epoch_end"])
        selection_min_delta = float(payload["best_selection_min_delta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint lacks valid candidate-selection metadata: {checkpoint_path}"
        ) from exc
    if candidate_start < 1 or candidate_end < candidate_start:
        raise ValueError(f"Checkpoint has an invalid candidate epoch range: {checkpoint_path}")
    try:
        configured_max_epoch = int(fusion_config["max_epoch"])
        configured_candidate_epochs = int(fusion_config["candidate_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint fusion_config lacks candidate epoch settings: {checkpoint_path}"
        ) from exc
    expected_start = max(1, configured_max_epoch - configured_candidate_epochs + 1)
    if candidate_start != expected_start or candidate_end != configured_max_epoch:
        raise ValueError(
            f"Checkpoint candidate range [{candidate_start}, {candidate_end}] does not match "
            f"fusion_config [{expected_start}, {configured_max_epoch}]: {checkpoint_path}"
        )
    if checkpoint_epoch != best_epoch:
        raise ValueError(
            f"Checkpoint epoch and best_epoch differ: epoch={checkpoint_epoch}, "
            f"best_epoch={best_epoch}: {checkpoint_path}"
        )
    if not candidate_start <= checkpoint_epoch <= candidate_end:
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} is outside candidate range "
            f"[{candidate_start}, {candidate_end}]: {checkpoint_path}"
        )
    if selection_min_delta != 0.0:
        raise ValueError(
            f"Candidate checkpoint must use raw monitor selection with min_delta=0.0: {checkpoint_path}"
        )
    return {
        "dataset_id": dataset_id,
        "seed": int(seed),
        "active_views": list(CORE_VIEWS),
        "epoch": checkpoint_epoch,
        "step": payload.get("step"),
        "selection_scope": selection_scope,
        "selection_rule": selection_rule,
        "candidate_epoch_start": candidate_start,
        "candidate_epoch_end": candidate_end,
        "best_selection_min_delta": selection_min_delta,
    }


def _load_checkpoint_manifest(
    path: Path,
    *,
    datasets: list[str],
    seeds: list[int],
    require_complete_grid: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Complete-only checkpoint manifest is missing: {path}")
    frame = pd.read_csv(path)
    required = {"dataset_id", "seed", "checkpoint_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Checkpoint manifest lacks columns: {sorted(missing)}")
    if frame.duplicated(["dataset_id", "seed"]).any():
        duplicates = frame.loc[frame.duplicated(["dataset_id", "seed"], keep=False), ["dataset_id", "seed"]]
        raise ValueError(f"Checkpoint manifest has duplicate dataset/seed entries:\n{duplicates.to_string(index=False)}")

    requested = {(dataset_id, int(seed)) for dataset_id in datasets for seed in seeds}
    if require_complete_grid:
        expected = {(dataset_id, seed) for dataset_id in DEFAULT_DATASETS for seed in DEFAULT_SEEDS}
        actual = {(str(row.dataset_id), int(row.seed)) for row in frame.itertuples(index=False)}
        if actual != expected:
            raise ValueError(
                f"Checkpoint manifest must exactly cover the standard ten runs; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    records: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoint_path_owners: dict[Path, tuple[str, int]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["dataset_id"]), int(row["seed"]))
        if key not in requested:
            continue
        raw_path = Path(str(row["checkpoint_path"])).expanduser()
        checkpoint_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
        checkpoint_path = checkpoint_path.resolve()
        previous_owner = checkpoint_path_owners.get(checkpoint_path)
        if previous_owner is not None and previous_owner != key:
            raise ValueError(
                f"A standard checkpoint cannot be reused for multiple dataset/seed entries: "
                f"{checkpoint_path} is assigned to {previous_owner} and {key}"
            )
        checkpoint_path_owners[checkpoint_path] = key
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Standard UMCL checkpoint is missing for {key}: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        identity = _validate_checkpoint_payload(
            checkpoint_path,
            payload,
            dataset_id=key[0],
            seed=key[1],
        )
        pretrained_model, pretrained_source = _infer_pretrained_model(
            checkpoint_path, row.get("pretrained_model")
        )
        records[key] = {
            **identity,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "model_source": "standard_umcl_checkpoint_manifest",
            "pretrained_model": pretrained_model,
            "pretrained_model_source": pretrained_source,
            "manifest_path": str(path),
        }
        del payload
    missing_requested = requested - set(records)
    if missing_requested:
        raise ValueError(f"Checkpoint manifest lacks requested dataset/seed entries: {sorted(missing_requested)}")
    return records


def _find_completed_run(
    run_root: Path,
    regime: str,
    dataset_id: str,
    seed: int,
    experiment_signature: str,
) -> dict[str, Any] | None:
    base = run_root / "runs" / regime / dataset_id / f"seed_{seed}"
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
            and len(payload.get("fixed_subset_tests") or {}) == len(SUBSET_SPECS)
        ):
            return payload
    return None


def _prepare_mixed_run(
    args: argparse.Namespace, dataset_id: str, seed: int
) -> tuple[argparse.Namespace, Any, dict[str, Any], dict[str, Any]]:
    run_args = copy.copy(args)
    config = resolve_fusion_config(dataset_id=dataset_id, mode="full")
    config.fusion_variant = "poe"
    config.active_views = ",".join(CORE_VIEWS)
    config.data_loader_seed = int(seed)
    config.resolved_overrides = {
        **config.resolved_overrides,
        "active_views": config.active_views,
        "data_loader_seed": seed,
        "fusion_variant": "poe",
    }
    run_args.batch_size = int(config.batch_size) if args.batch_size is None else int(args.batch_size)
    run_args.max_epoch = int(config.max_epoch) if args.max_epoch is None else int(args.max_epoch)
    run_args.candidate_epochs = (
        int(config.candidate_epochs) if args.candidate_epochs is None else int(args.candidate_epochs)
    )
    run_args.early_stop_patience = (
        int(config.early_stop_patience) if args.early_stop_patience is None else int(args.early_stop_patience)
    )
    for name in ("batch_size", "max_epoch", "candidate_epochs", "early_stop_patience"):
        if int(getattr(run_args, name)) < 1:
            raise ValueError(f"{name} must be at least 1.")
    config.max_epoch = run_args.max_epoch
    config.candidate_epochs = run_args.candidate_epochs
    loss_overrides = {
        "view_reconstruction_loss_weight": args.reconstruction_loss_weight,
        "primary_task_loss_weight": args.task_primary_loss_weight,
        "stance_auxiliary_loss_weight": args.aux_stance_loss_weight,
        "posterior_kl_loss_weight": args.kl_loss_weight,
    }
    for key, value in loss_overrides.items():
        if value is not None:
            setattr(config, key, float(value))
            config.resolved_overrides = {**config.resolved_overrides, key: float(value)}
    optimizer_config = _resolve_optimizer_config(run_args, config)
    config = apply_optimizer_config(config, optimizer_config)
    apply_sample_latent_default(config, list(CORE_VIEWS))
    sampling = {
        "training": DeterministicMixedSubsetMasker(
            seed=seed, phase="train", vary_by_epoch=True
        ).manifest(),
        "validation": DeterministicMixedSubsetMasker(
            seed=seed, phase="validation", vary_by_epoch=False
        ).manifest(),
    }
    signature_payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_id": dataset_id,
        "regime": "mixed",
        "seed": seed,
        "fusion_config": asdict(config),
        "batch_size": run_args.batch_size,
        "max_epoch": run_args.max_epoch,
        "candidate_epochs": run_args.candidate_epochs,
        "selection_scope": BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
        "selection_rule": BEST_SELECTION_RULE_RAW_MAX,
        "candidate_epoch_start": max(
            1, run_args.max_epoch - run_args.candidate_epochs + 1
        ),
        "candidate_epoch_end": run_args.max_epoch,
        "max_steps": run_args.max_steps,
        "early_stop_patience": run_args.early_stop_patience,
        "early_stop_min_delta": run_args.early_stop_min_delta,
        "train_splits": parse_split_arg(run_args.splits),
        "eval_splits": parse_split_arg(run_args.eval_splits),
        "subset_sampling": sampling,
        "loader": {
            "num_workers": run_args.num_workers,
            "prefetch_factor": run_args.prefetch_factor,
            "disable_persistent_workers": run_args.disable_persistent_workers,
            "disable_pin_memory": run_args.disable_pin_memory,
        },
        "device": run_args.device,
        "cuda_index": run_args.cuda_index,
        "amp": run_args.amp,
    }
    return run_args, config, optimizer_config, signature_payload


def _validation_rows(masker: DeterministicMixedSubsetMasker | None) -> list[dict[str, Any]]:
    if masker is None:
        return []
    rows: list[dict[str, Any]] = []
    for raw in masker.recorded_assignment_rows():
        rows.append(
            {
                **raw,
                "requested_views": "|".join(raw["requested_views"]),
                "observed_views": "|".join(raw["observed_views"]),
                "intrinsically_available_views": "|".join(raw["intrinsically_available_views"]),
            }
        )
    return rows


def _evaluate_fixed_subsets(
    *,
    model: torch.nn.Module,
    loader: Any,
    primary_task_spec: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    dataset_id: str,
    regime: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixed_subset_tests: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    for subset_name, active_views in SUBSET_SPECS:
        result = _run_masked_test(
            model=model,
            loader=loader,
            primary_task_spec=primary_task_spec,
            device=device,
            use_amp=use_amp,
            view_mask=ViewMask(active_views=set(active_views)),
            collect_uncertainty=False,
        )
        fixed_subset_tests[subset_name] = result
        test_rows.append(
            {
                "dataset_id": dataset_id,
                "regime": regime,
                "seed": seed,
                "subset": subset_name,
                "active_views": "|".join(active_views),
                "event_count": result["event_count"],
                "valid_prediction_count": result["valid_prediction_count"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
            }
        )
    return fixed_subset_tests, test_rows


def _run_mixed(args: argparse.Namespace, dataset_id: str, seed: int) -> dict[str, Any]:
    regime = "mixed"
    run_args, config, optimizer_config, signature_payload = _prepare_mixed_run(args, dataset_id, seed)
    experiment_signature = _signature(signature_payload)
    run_root = Path(args.run_root)
    if not args.force:
        completed = _find_completed_run(run_root, regime, dataset_id, seed, experiment_signature)
        if completed is not None:
            print(f"Skipping completed run: dataset={dataset_id} regime={regime} seed={seed}")
            return completed

    started_at = _now_iso()
    started_perf = time.perf_counter()
    run_dir = run_root / "runs" / regime / dataset_id / f"seed_{seed}" / _now_dirstamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = _resolve_device(args.device, args.cuda_index)
    use_amp = _resolve_amp(args.amp, device)
    save_selected_checkpoint = _resolve_checkpoint_choice(args.checkpoint)
    train_splits = parse_split_arg(args.splits)
    eval_splits = parse_split_arg(args.eval_splits)
    full_mask = ViewMask(active_views=set(CORE_VIEWS))
    seed_everything(seed)

    train_masker = DeterministicMixedSubsetMasker(
        seed=seed, phase="train", vary_by_epoch=True, record_assignments=False
    )
    eval_masker = DeterministicMixedSubsetMasker(
        seed=seed, phase="validation", vary_by_epoch=False, record_assignments=True
    )
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
    write_json(
        run_dir / "mask_reproducibility.json",
        {
            "regime": regime,
            "training": train_masker.manifest(),
            "validation": eval_masker.manifest(),
            "validation_assignments_file": "validation_mask_assignments.csv",
        },
    )
    artifact_results = ensure_model_inputs(
        dataset_id=dataset_id,
        mode="full",
        active_views=list(CORE_VIEWS),
        config=config,
        data_root=data_root,
        input_root=data_root / "preprocessed",
        options=EnsureArtifactsOptions(),
    )
    print(f"Running dataset={dataset_id} regime={regime} seed={seed} device={device}")
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
            required_views=set(CORE_VIEWS),
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
        required_views=set(CORE_VIEWS),
    )
    if eval_dims != feature_dims or eval_task_spec != task_spec:
        raise ValueError("Training and validation feature/task specifications differ.")

    model = create_fusion_model(
        config, feature_dims, primary_task_spec=task_spec, active_views=set(CORE_VIEWS)
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
        optimizer, **_scheduler_params(optimizer_config)
    )
    candidate_start = max(1, run_args.max_epoch - run_args.candidate_epochs + 1)
    selected: dict[str, Any] = {
        "epoch": None,
        "step": None,
        "monitor_value": None,
        "state_dict": None,
    }

    def on_candidate_best(event: dict[str, Any]) -> None:
        record = event["epoch_record"]
        selected["epoch"] = int(record.epoch)
        selected["step"] = int(event["step"])
        selected["monitor_value"] = float(record.monitor_value)
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
        on_new_best=on_candidate_best,
        enable_early_stop=False,
        train_subset_masker=train_masker,
        eval_subset_masker=eval_masker,
        best_selection_start_epoch=candidate_start,
        best_selection_min_delta=0.0,
        best_selection_scope=BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
        best_selection_rule=BEST_SELECTION_RULE_RAW_MAX,
    )
    if selected["state_dict"] is None:
        raise ValueError(
            f"No selectable epoch was reached; candidate range starts at epoch {candidate_start}."
        )
    if int(training_result["best_epoch"]) != int(selected["epoch"]):
        raise RuntimeError(
            "Mixed-subset candidate-best state does not match the shared training result: "
            f"selected={selected['epoch']} training={training_result['best_epoch']}."
        )
    model.load_state_dict(selected["state_dict"])
    del selected["state_dict"]

    checkpoint_path: Path | None = None
    if save_selected_checkpoint:
        checkpoint_path = run_dir / "checkpoint_best.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            primary_task_spec=task_spec,
            step=int(selected["step"]),
            epoch=int(selected["epoch"]),
            extra_state={
                "experiment": "unified_mixed_subset",
                "regime": regime,
                "active_views": list(CORE_VIEWS),
                "selection_monitor_value": selected["monitor_value"],
                "best_epoch": int(selected["epoch"]),
                "best_monitor_value": selected["monitor_value"],
                "selection_scope": BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                "selection_rule": BEST_SELECTION_RULE_RAW_MAX,
                "candidate_epoch_start": candidate_start,
                "candidate_epoch_end": run_args.max_epoch,
                "best_selection_min_delta": 0.0,
                "mask_reproducibility": "mask_reproducibility.json",
            },
        )

    validation_rows = _validation_rows(eval_masker)
    if validation_rows:
        _write_csv(
            run_dir / "validation_mask_assignments.csv",
            list(validation_rows[0]),
            validation_rows,
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
            required_views=set(CORE_VIEWS),
        )
    )
    if test_dims != feature_dims or test_task_spec != task_spec:
        raise ValueError("Training and test feature/task specifications differ.")

    fixed_subset_tests, test_rows = _evaluate_fixed_subsets(
        model=model,
        loader=test_loader,
        primary_task_spec=task_spec,
        device=device,
        use_amp=use_amp,
        dataset_id=dataset_id,
        regime=regime,
        seed=seed,
    )
    _write_csv(run_dir / "fixed_subset_test_results.csv", list(test_rows[0]), test_rows)
    write_json(run_dir / "fixed_subset_test_results.json", {"fixed_subset_tests": fixed_subset_tests})

    ended_at = _now_iso()
    summary = {
        "status": "complete",
        "experiment_signature": experiment_signature,
        "experiment_version": EXPERIMENT_VERSION,
        "run": {
            "dataset_id": dataset_id,
            "regime": regime,
            "seed": seed,
            "run_dir": str(run_dir),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": max(0.0, time.perf_counter() - started_perf),
        },
        "selection": {
            "epoch": selected["epoch"],
            "step": selected["step"],
            "monitor_value": selected["monitor_value"],
            "candidate_epoch_start": candidate_start,
            "candidate_epoch_end": run_args.max_epoch,
            "selection_scope": BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
            "selection_rule": BEST_SELECTION_RULE_RAW_MAX,
            "best_selection_min_delta": 0.0,
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
        "training": {"performed": True, **_jsonable(training_result)},
        "fixed_subset_tests": fixed_subset_tests,
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


def _checkpoint_config(payload: dict[str, Any], dataset_id: str) -> Any:
    checkpoint_config = dict(payload["fusion_config"])
    checkpoint_mode = str(checkpoint_config.get("mode", "full"))
    if checkpoint_mode != "full":
        raise ValueError(f"Complete-only checkpoint must use full mode, got {checkpoint_mode!r}.")
    config = resolve_fusion_config(dataset_id=dataset_id, mode="full")
    for key, value in checkpoint_config.items():
        if key in {"dataset_id", "mode", "resolved_overrides"}:
            continue
        if hasattr(config, key):
            setattr(config, key, copy.deepcopy(value))
    config.dataset_id = dataset_id
    config.mode = "full"
    config.fusion_variant = "poe"
    config.active_views = ",".join(CORE_VIEWS)
    config = apply_optimizer_config(config, normalize_optimizer_config(config.optimizer))
    apply_sample_latent_default(config, list(CORE_VIEWS))
    return config


def _validate_task_compatibility(
    checkpoint_task_spec: dict[str, Any],
    dataset_task_spec: dict[str, Any],
) -> None:
    if checkpoint_task_spec.get("task_name") != dataset_task_spec.get("task_name"):
        raise ValueError("Checkpoint primary task name does not match the test dataset.")
    checkpoint_classes = list(checkpoint_task_spec.get("class_names") or [])
    dataset_classes = list(dataset_task_spec.get("class_names") or [])
    if checkpoint_classes != dataset_classes:
        raise ValueError("Checkpoint class names do not match the test dataset.")


def _preflight_checkpoint_runtime(
    args: argparse.Namespace,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any]:
    """Strictly validate one standard checkpoint against its exact test dataset on CPU."""
    dataset_id = str(checkpoint_record["dataset_id"])
    seed = int(checkpoint_record["seed"])
    checkpoint_path = Path(checkpoint_record["checkpoint_path"])
    data_root = Path(args.data_root)
    override_token = set_pretrained_model_override(checkpoint_record["pretrained_model"])
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        _validate_checkpoint_payload(
            checkpoint_path,
            payload,
            dataset_id=dataset_id,
            seed=seed,
        )
        config = _checkpoint_config(payload, dataset_id)
        batch_size = int(config.batch_size) if args.batch_size is None else int(args.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        ensure_model_inputs(
            dataset_id=dataset_id,
            mode="full",
            active_views=list(CORE_VIEWS),
            config=config,
            data_root=data_root,
            input_root=data_root / "preprocessed",
            options=EnsureArtifactsOptions(),
        )
        loader, sample_count, selected_splits, feature_dims, _, task_spec = build_eval_loader(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=batch_size,
            splits=STRICT_TEST_SPLITS,
            device=torch.device("cpu"),
            seed=seed,
            num_workers_override=args.num_workers,
            prefetch_factor_override=args.prefetch_factor,
            persistent_workers_override=(False if args.disable_persistent_workers else None),
            pin_memory_override=False,
            required_views=set(CORE_VIEWS),
        )
        if selected_splits != STRICT_TEST_SPLITS:
            raise ValueError(
                f"Complete-only evaluation requires exact test split, got {selected_splits}."
            )
        _validate_task_compatibility(dict(payload["primary_task_spec"]), task_spec)
        model = create_fusion_model(
            config,
            feature_dims,
            primary_task_spec=task_spec,
            active_views=set(CORE_VIEWS),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del model, loader, payload
        return {
            "dataset_id": dataset_id,
            "seed": seed,
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
            "test_splits": selected_splits,
            "test_sample_count": sample_count,
            "feature_dims": feature_dims,
            "primary_task_spec": task_spec,
            "strict_state_dict_load": True,
        }
    finally:
        reset_pretrained_model_override(override_token)


def _reference_comparison(
    fixed_subset_tests: dict[str, Any],
    specialist_reference: dict[str, Any],
    *,
    dataset_id: str,
    seed: int,
) -> dict[str, Any]:
    measured = fixed_subset_tests["C+S+P"]
    macro_f1_delta = float(measured["macro_f1"]) - float(specialist_reference["macro_f1"])
    accuracy_delta = float(measured["accuracy"]) - float(specialist_reference["accuracy"])
    warning = (
        abs(macro_f1_delta) > REFERENCE_WARNING_TOLERANCE
        or abs(accuracy_delta) > REFERENCE_WARNING_TOLERANCE
    )
    if warning:
        print(
            "WARNING: complete-only C+S+P inference differs from the specialist reference "
            f"for dataset={dataset_id} seed={seed}: "
            f"macro_f1_delta={macro_f1_delta:.8f}, accuracy_delta={accuracy_delta:.8f}"
        )
    return {
        "reference_source": specialist_reference["run_dir"],
        "reference_macro_f1": specialist_reference["macro_f1"],
        "measured_macro_f1": measured["macro_f1"],
        "macro_f1_delta": macro_f1_delta,
        "reference_accuracy": specialist_reference["accuracy"],
        "measured_accuracy": measured["accuracy"],
        "accuracy_delta": accuracy_delta,
        "warning_tolerance": REFERENCE_WARNING_TOLERANCE,
        "warning": warning,
    }


def _run_complete_only(
    args: argparse.Namespace,
    dataset_id: str,
    seed: int,
    checkpoint_record: dict[str, Any],
    specialist_reference: dict[str, Any],
) -> dict[str, Any]:
    regime = "complete_only"
    checkpoint_path = Path(checkpoint_record["checkpoint_path"])
    signature_payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_id": dataset_id,
        "regime": regime,
        "seed": seed,
        "protocol": "reuse_candidate_best_csp_checkpoint_then_mask_at_test",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "checkpoint_epoch": checkpoint_record.get("epoch"),
        "checkpoint_step": checkpoint_record.get("step"),
        "selection_scope": checkpoint_record["selection_scope"],
        "selection_rule": checkpoint_record["selection_rule"],
        "candidate_epoch_start": checkpoint_record["candidate_epoch_start"],
        "candidate_epoch_end": checkpoint_record["candidate_epoch_end"],
        "model_source": checkpoint_record["model_source"],
        "pretrained_model": checkpoint_record["pretrained_model"],
        "pretrained_model_source": checkpoint_record["pretrained_model_source"],
        "test_splits": STRICT_TEST_SPLITS,
        "batch_size_override": args.batch_size,
        "loader": {
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "disable_persistent_workers": args.disable_persistent_workers,
            "disable_pin_memory": args.disable_pin_memory,
        },
        "device": args.device,
        "cuda_index": args.cuda_index,
        "amp": args.amp,
    }
    experiment_signature = _signature(signature_payload)
    run_root = Path(args.run_root)
    if not args.force:
        completed = _find_completed_run(run_root, regime, dataset_id, seed, experiment_signature)
        if completed is not None:
            print(f"Skipping completed run: dataset={dataset_id} regime={regime} seed={seed}")
            return completed

    started_at = _now_iso()
    started_perf = time.perf_counter()
    run_dir = run_root / "runs" / regime / dataset_id / f"seed_{seed}" / _now_dirstamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, args.cuda_index)
    use_amp = _resolve_amp(args.amp, device)
    data_root = Path(args.data_root)
    source_checkpoint = {
        **checkpoint_record,
        "trained_by_this_experiment": False,
    }
    write_json(
        run_dir / "run_config.json",
        {
            "status": "running",
            "experiment_signature": experiment_signature,
            "signature_payload": signature_payload,
            "source_checkpoint": source_checkpoint,
            "run_dir": str(run_dir),
            "started_at": started_at,
        },
    )

    override_token = set_pretrained_model_override(checkpoint_record["pretrained_model"])
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        _validate_checkpoint_payload(
            checkpoint_path,
            payload,
            dataset_id=dataset_id,
            seed=seed,
        )
        config = _checkpoint_config(payload, dataset_id)
        batch_size = int(config.batch_size) if args.batch_size is None else int(args.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        artifact_results = ensure_model_inputs(
            dataset_id=dataset_id,
            mode="full",
            active_views=list(CORE_VIEWS),
            config=config,
            data_root=data_root,
            input_root=data_root / "preprocessed",
            options=EnsureArtifactsOptions(),
        )
        test_loader, test_count, selected_test_splits, feature_dims, test_loader_settings, task_spec = build_eval_loader(
            dataset_id=dataset_id,
            config=config,
            data_root=data_root,
            batch_size=batch_size,
            splits=STRICT_TEST_SPLITS,
            device=device,
            seed=seed,
            num_workers_override=args.num_workers,
            prefetch_factor_override=args.prefetch_factor,
            persistent_workers_override=(False if args.disable_persistent_workers else None),
            pin_memory_override=(False if args.disable_pin_memory else None),
            required_views=set(CORE_VIEWS),
        )
        if selected_test_splits != STRICT_TEST_SPLITS:
            raise ValueError(
                f"Complete-only evaluation requires exact test split, got {selected_test_splits}."
            )
        _validate_task_compatibility(dict(payload["primary_task_spec"]), task_spec)
        model = create_fusion_model(
            config,
            feature_dims,
            primary_task_spec=task_spec,
            active_views=set(CORE_VIEWS),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to(device)
        del payload
        fixed_subset_tests, test_rows = _evaluate_fixed_subsets(
            model=model,
            loader=test_loader,
            primary_task_spec=task_spec,
            device=device,
            use_amp=use_amp,
            dataset_id=dataset_id,
            regime=regime,
            seed=seed,
        )
    finally:
        reset_pretrained_model_override(override_token)

    _write_csv(run_dir / "fixed_subset_test_results.csv", list(test_rows[0]), test_rows)
    write_json(run_dir / "fixed_subset_test_results.json", {"fixed_subset_tests": fixed_subset_tests})
    reference_comparison = _reference_comparison(
        fixed_subset_tests,
        specialist_reference,
        dataset_id=dataset_id,
        seed=seed,
    )
    write_json(run_dir / "csp_specialist_comparison.json", reference_comparison)
    ended_at = _now_iso()
    summary = {
        "status": "complete",
        "experiment_signature": experiment_signature,
        "experiment_version": EXPERIMENT_VERSION,
        "run": {
            "dataset_id": dataset_id,
            "regime": regime,
            "seed": seed,
            "run_dir": str(run_dir),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": max(0.0, time.perf_counter() - started_perf),
        },
        "protocol": {
            "definition": (
                "reuse candidate-epoch-best standard C+S+P UMCL checkpoint and apply "
                "seven fixed masks at test time"
            ),
            "test_splits": selected_test_splits,
            "parameter_updates": 0,
        },
        "source_checkpoint": source_checkpoint,
        "training": {"performed": False},
        "selection": {
            "performed_by_this_experiment": False,
            "source_epoch": checkpoint_record.get("epoch"),
            "source_step": checkpoint_record.get("step"),
            "selection_scope": checkpoint_record["selection_scope"],
            "selection_rule": checkpoint_record["selection_rule"],
            "candidate_epoch_start": checkpoint_record["candidate_epoch_start"],
            "candidate_epoch_end": checkpoint_record["candidate_epoch_end"],
        },
        "data": {
            "train_splits": None,
            "eval_splits": None,
            "test_splits": selected_test_splits,
            "train_sample_count": None,
            "eval_sample_count": None,
            "test_sample_count": test_count,
            "feature_dims": feature_dims,
            "primary_task_spec": task_spec,
            "test_loader_settings": test_loader_settings,
            "artifacts": _jsonable(artifact_results),
        },
        "signature_payload": signature_payload,
        "reference_comparison": reference_comparison,
        "fixed_subset_tests": fixed_subset_tests,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "run_config.json",
        {
            "status": "complete",
            "experiment_signature": experiment_signature,
            "signature_payload": signature_payload,
            "source_checkpoint": source_checkpoint,
            "run_dir": str(run_dir),
            "started_at": started_at,
            "ended_at": ended_at,
        },
    )
    return summary


def _specialist_seed_rows(path: Path, datasets: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Subset-specific result file is missing: {path}")
    frame = pd.read_csv(path)
    required = {"dataset_id", "seed", "active_views", "test_macro_f1", "test_accuracy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Subset-specific results lack columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        if record["dataset_id"] not in datasets:
            continue
        subset = ACTIVE_VIEWS_TO_SUBSET.get(str(record["active_views"]))
        if subset is None:
            continue
        rows.append(
            {
                "dataset_id": record["dataset_id"],
                "regime": "subset_specific",
                "seed": int(record["seed"]),
                "subset": subset,
                "active_views": record["active_views"],
                "macro_f1": float(record["test_macro_f1"]),
                "accuracy": float(record["test_accuracy"]),
                "run_dir": "results/result_view_contribution.csv",
            }
        )
    return rows


def _validate_specialist_inputs(
    rows: list[dict[str, Any]],
    *,
    datasets: list[str],
    seeds: list[int],
    require_complete_grid: bool,
) -> None:
    expected_datasets = list(DEFAULT_DATASETS) if require_complete_grid else datasets
    expected_seeds = list(DEFAULT_SEEDS) if require_complete_grid else seeds
    expected = {
        (dataset_id, int(seed), subset)
        for dataset_id in expected_datasets
        for seed in expected_seeds
        for subset in SUBSET_ORDER
    }
    expected_seed_set = set(expected_seeds)
    selected = [
        row for row in rows
        if row["dataset_id"] in expected_datasets and int(row["seed"]) in expected_seed_set
    ]
    actual_keys = [(row["dataset_id"], int(row["seed"]), row["subset"]) for row in selected]
    duplicate_keys = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"Subset-specific results contain duplicate dataset/seed/subset rows: {duplicate_keys}")
    actual = set(actual_keys)
    if actual != expected:
        raise ValueError(
            f"Subset-specific results do not cover the required grid; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _preflight_inputs(
    args: argparse.Namespace,
    datasets: list[str],
    regimes: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    specialist_path = Path(args.specialist_results).expanduser().resolve()
    specialist_rows = _specialist_seed_rows(specialist_path, datasets)
    _validate_specialist_inputs(
        specialist_rows,
        datasets=datasets,
        seeds=seeds,
        require_complete_grid=args.require_complete_grid,
    )
    checkpoint_records: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoint_runtime_validations: list[dict[str, Any]] = []
    if "complete_only" in regimes:
        checkpoint_records = _load_checkpoint_manifest(
            Path(args.complete_only_checkpoint_manifest),
            datasets=datasets,
            seeds=seeds,
            require_complete_grid=args.require_complete_grid,
        )
        for key in sorted(checkpoint_records):
            print(f"Preflighting standard checkpoint: dataset={key[0]} seed={key[1]}")
            validation = _preflight_checkpoint_runtime(args, checkpoint_records[key])
            checkpoint_records[key]["runtime_validation"] = validation
            checkpoint_runtime_validations.append(validation)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "preflight.json",
        {
            "experiment_version": EXPERIMENT_VERSION,
            "datasets": datasets,
            "regimes": regimes,
            "seeds": seeds,
            "specialist_results": str(specialist_path),
            "specialist_rows_validated": len(specialist_rows),
            "complete_only_checkpoints": [
                checkpoint_records[key] for key in sorted(checkpoint_records)
            ],
            "complete_only_runtime_validations": checkpoint_runtime_validations,
            "status": "complete",
        },
    )
    return specialist_rows, checkpoint_records


def _specialist_csp_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["dataset_id"], int(row["seed"])): row
        for row in rows
        if row["subset"] == "C+S+P"
    }


def _trained_seed_rows(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        run = summary["run"]
        for subset_name, _ in SUBSET_SPECS:
            result = summary["fixed_subset_tests"][subset_name]
            rows.append(
                {
                    "dataset_id": run["dataset_id"],
                    "regime": run["regime"],
                    "seed": int(run["seed"]),
                    "subset": subset_name,
                    "active_views": "|".join(SUBSET_TO_VIEWS[subset_name]),
                    "macro_f1": float(result["macro_f1"]),
                    "accuracy": float(result["accuracy"]),
                    "run_dir": run["run_dir"],
                }
            )
    return rows


def _aggregate_rows(seed_rows: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for regime in REGIME_LABELS:
            for subset in SUBSET_ORDER:
                selected = [
                    row for row in seed_rows
                    if row["dataset_id"] == dataset_id
                    and row["regime"] == regime
                    and row["subset"] == subset
                ]
                if not selected:
                    continue
                f1_mean, f1_std, f1_n = _mean_and_sample_std(row["macro_f1"] for row in selected)
                acc_mean, acc_std, acc_n = _mean_and_sample_std(row["accuracy"] for row in selected)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "regime": regime,
                        "regime_label": REGIME_LABELS[regime],
                        "subset": subset,
                        "active_views": "|".join(SUBSET_TO_VIEWS[subset]),
                        "macro_f1_mean": f1_mean,
                        "macro_f1_sample_std": f1_std,
                        "macro_f1_seed_count": f1_n,
                        "accuracy_mean": acc_mean,
                        "accuracy_sample_std": acc_std,
                        "accuracy_seed_count": acc_n,
                    }
                )
    return rows


def _summary_rows(aggregate_rows: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for regime in REGIME_LABELS:
            selected = [
                row for row in aggregate_rows
                if row["dataset_id"] == dataset_id and row["regime"] == regime
            ]
            if len(selected) != len(SUBSET_ORDER):
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "regime": regime,
                    "regime_label": REGIME_LABELS[regime],
                    "mean_subset_macro_f1": statistics.fmean(float(row["macro_f1_mean"]) for row in selected),
                    "mean_subset_accuracy": statistics.fmean(float(row["accuracy_mean"]) for row in selected),
                    "subset_count": len(selected),
                }
            )
    return rows


def _gap_rows(aggregate_rows: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset_id"], row["regime"], row["subset"]): row for row in aggregate_rows
    }
    rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        dataset_gaps: list[float] = []
        for subset in SUBSET_ORDER:
            unified = lookup.get((dataset_id, "mixed", subset))
            specialist = lookup.get((dataset_id, "subset_specific", subset))
            if unified is None or specialist is None:
                continue
            gap = float(unified["macro_f1_mean"]) - float(specialist["macro_f1_mean"])
            dataset_gaps.append(gap)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "unified_macro_f1_mean": unified["macro_f1_mean"],
                    "specialist_macro_f1_mean": specialist["macro_f1_mean"],
                    "signed_gap_unified_minus_specialist": gap,
                    "absolute_gap": abs(gap),
                    "mean_absolute_gap_across_subsets": None,
                }
            )
        if dataset_gaps:
            mean_absolute_gap = statistics.fmean(abs(value) for value in dataset_gaps)
            for row in rows:
                if row["dataset_id"] == dataset_id:
                    row["mean_absolute_gap_across_subsets"] = mean_absolute_gap
    return rows


def _latex_table(
    aggregate_rows: list[dict[str, Any]], datasets: list[str], metric: str, caption: str, label: str
) -> str:
    value_key = f"{metric}_mean"
    lookup = {
        (row["dataset_id"], row["regime"], row["subset"]): row.get(value_key)
        for row in aggregate_rows
    }
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{llrrrrrrrr}",
        r"    \hline",
        "    Dataset & Training regime & C & S & P & C+S & C+P & S+P & C+S+P & Mean \\\\",
        r"    \hline",
    ]
    for dataset_id in datasets:
        for regime in REGIME_LABELS:
            values = [lookup.get((dataset_id, regime, subset)) for subset in SUBSET_ORDER]
            formatted = ["N/A" if value is None else f"{100.0 * float(value):.2f}" for value in values]
            mean_value = None if any(value is None for value in values) else statistics.fmean(float(v) for v in values)
            formatted_mean = "N/A" if mean_value is None else f"{100.0 * mean_value:.2f}"
            lines.append(
                f"    {dataset_id} & {REGIME_LABELS[regime]} & "
                + " & ".join([*formatted, formatted_mean])
                + " \\\\")
        lines.append(r"    \hline")
    lines.extend([r"    \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _validate_complete_grid(
    aggregate_rows: list[dict[str, Any]], datasets: list[str], seeds: list[int]
) -> None:
    if datasets != list(DEFAULT_DATASETS) or sorted(seeds) != sorted(DEFAULT_SEEDS):
        raise ValueError("--require_complete_grid requires the default datasets and seeds.")
    lookup = {(row["dataset_id"], row["regime"], row["subset"]): row for row in aggregate_rows}
    missing: list[str] = []
    for dataset_id in datasets:
        for regime in REGIME_LABELS:
            for subset in SUBSET_ORDER:
                row = lookup.get((dataset_id, regime, subset))
                if row is None or int(row["macro_f1_seed_count"]) != len(DEFAULT_SEEDS):
                    missing.append(f"{dataset_id}/{regime}/{subset}")
    if missing:
        raise ValueError(f"Incomplete five-seed aggregate grid: {missing}")


def _validate_seed_grid(seed_rows: list[dict[str, Any]], datasets: list[str], seeds: list[int]) -> None:
    expected_seeds = set(DEFAULT_SEEDS)
    if datasets != list(DEFAULT_DATASETS) or set(seeds) != expected_seeds:
        raise ValueError("Complete seed-grid validation requires the default datasets and seeds.")
    invalid: list[str] = []
    for dataset_id in datasets:
        for regime in REGIME_LABELS:
            for subset in SUBSET_ORDER:
                selected = [
                    row for row in seed_rows
                    if row["dataset_id"] == dataset_id
                    and row["regime"] == regime
                    and row["subset"] == subset
                ]
                actual_seeds = [int(row["seed"]) for row in selected]
                if len(actual_seeds) != len(expected_seeds) or set(actual_seeds) != expected_seeds:
                    invalid.append(
                        f"{dataset_id}/{regime}/{subset}: seeds={sorted(actual_seeds)}"
                    )
    if invalid:
        raise ValueError(f"Incomplete or duplicate per-seed result grid: {invalid}")


def _write_aggregate_outputs(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    datasets: list[str],
    seeds: list[int],
    *,
    validate_complete: bool = False,
) -> None:
    run_root = Path(args.run_root)
    seed_rows = _specialist_seed_rows(Path(args.specialist_results), datasets)
    seed_rows.extend(_trained_seed_rows(summaries))
    dataset_rank = {name: index for index, name in enumerate(datasets)}
    regime_rank = {name: index for index, name in enumerate(REGIME_LABELS)}
    subset_rank = {name: index for index, name in enumerate(SUBSET_ORDER)}
    seed_rows.sort(
        key=lambda row: (
            dataset_rank[row["dataset_id"]],
            regime_rank[row["regime"]],
            int(row["seed"]),
            subset_rank[row["subset"]],
        )
    )
    aggregate_rows = _aggregate_rows(seed_rows, datasets)
    summary_rows = _summary_rows(aggregate_rows, datasets)
    gap_rows = _gap_rows(aggregate_rows, datasets)
    if seed_rows:
        _write_csv(run_root / "subset_seed_results.csv", list(seed_rows[0]), seed_rows)
    if aggregate_rows:
        _write_csv(run_root / "subset_aggregate.csv", list(aggregate_rows[0]), aggregate_rows)
    if summary_rows:
        _write_csv(run_root / "regime_summary.csv", list(summary_rows[0]), summary_rows)
    if gap_rows:
        _write_csv(run_root / "unified_specialist_gap.csv", list(gap_rows[0]), gap_rows)
    (run_root / "macro_f1_main_table.tex").write_text(
        _latex_table(
            aggregate_rows,
            datasets,
            "macro_f1",
            r"Unified inference under mixed observed-view subsets (Macro-F1, \%).",
            "tab:unified-mixed-subset-macro-f1",
        ),
        encoding="utf-8",
    )
    for dataset_id in datasets:
        (run_root / f"{dataset_id.lower()}_macro_f1_main_table.tex").write_text(
            _latex_table(
                aggregate_rows,
                [dataset_id],
                "macro_f1",
                rf"Unified inference under mixed observed-view subsets on {dataset_id} (Macro-F1, \%).",
                f"tab:unified-mixed-subset-{dataset_id.lower()}-macro-f1",
            ),
            encoding="utf-8",
        )
    (run_root / "accuracy_supplementary_table.tex").write_text(
        _latex_table(
            aggregate_rows,
            datasets,
            "accuracy",
            r"Unified inference under mixed observed-view subsets (accuracy, \%).",
            "tab:unified-mixed-subset-accuracy",
        ),
        encoding="utf-8",
    )
    if args.require_complete_grid and validate_complete:
        _validate_seed_grid(seed_rows, datasets, seeds)
        _validate_complete_grid(aggregate_rows, datasets, seeds)
    complete_for_plot = all(
        any(
            row["dataset_id"] == dataset_id
            and row["regime"] == regime
            and row["subset"] == subset
            for row in aggregate_rows
        )
        for dataset_id in datasets
        for regime in REGIME_LABELS
        for subset in SUBSET_ORDER
    )
    if complete_for_plot:
        from plot.scripts.plot_unified_mixed_subset import generate_figure

        generate_figure(
            input_csv=run_root / "subset_aggregate.csv",
            output_pdf=run_root / "unified_mixed_subset.pdf",
        )


def main() -> None:
    args = parse_args()
    datasets, regimes, seeds = _validate_grid(args)
    specialist_rows, checkpoint_records = _preflight_inputs(args, datasets, regimes, seeds)
    specialist_csp = _specialist_csp_lookup(specialist_rows)
    summaries: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for regime in regimes:
            for seed in seeds:
                if regime == "complete_only":
                    summaries.append(
                        _run_complete_only(
                            args,
                            dataset_id,
                            seed,
                            checkpoint_records[(dataset_id, seed)],
                            specialist_csp[(dataset_id, seed)],
                        )
                    )
                else:
                    summaries.append(_run_mixed(args, dataset_id, seed))
                _write_aggregate_outputs(args, summaries, datasets, seeds, validate_complete=False)
    _write_aggregate_outputs(args, summaries, datasets, seeds, validate_complete=True)
    print(f"Completed {len(summaries)} configurations. Results: {Path(args.run_root)}")


if __name__ == "__main__":
    main()
