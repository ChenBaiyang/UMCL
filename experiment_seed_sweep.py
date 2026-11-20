"""UMCL experiment: seed sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

from config.resolver import resolve_encoder_config
from models.fusion.training import (
    BEST_SELECTION_RULE_RAW_MAX,
    BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
)


SEEDS = (2, 4, 8, 16, 32)
MANIFEST_FIELDS = ("dataset_id", "seed", "checkpoint_path", "pretrained_model")
STANDARD_ACTIVE_VIEWS = ("content", "stance", "propagation")
REPOSITORY_ROOT = Path(__file__).resolve().parent
TRAIN_ENTRYPOINT = REPOSITORY_ROOT / "train_umcl.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy train_umcl.py entrypoint for the five standard seeds and "
            "register candidate-epoch-best checkpoints."
        )
    )
    parser.add_argument("--dataset_id", required=True, help="Single dataset id, e.g. PHEME.")
    parser.add_argument(
        "--pretrained_model",
        default=None,
        help="Optional HuggingFace model id or local model path passed to train_umcl.py.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_root", default="runs")
    parser.add_argument(
        "--manifest",
        default="results/standard_umcl_checkpoints.csv",
        help="Checkpoint manifest updated after each successful seed.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _absolute_from_repository_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _resolve_manifest_checkpoint(manifest_path: Path, raw_value: str) -> Path:
    checkpoint_path = Path(str(raw_value).strip()).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = manifest_path.parent / checkpoint_path
    return checkpoint_path.resolve()


def _artifact_compatible_data_root_argument(
    *,
    requested_value: str | Path,
    resolved_data_root: Path,
    dataset_id: str,
) -> str:
    """Reuse the path spelling stored in event-index metadata when equivalent."""
    metadata_path = (
        resolved_data_root
        / "content_view"
        / dataset_id
        / "full"
        / "artifact_metadata.json"
    )
    if not metadata_path.is_file():
        return str(requested_value)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid artifact metadata JSON: {metadata_path}") from exc
    event_entry = ((metadata.get("artifacts") or {}).get("event_index") or {})
    fingerprint_input = event_entry.get("fingerprint_input") or {}
    stored_input_root_value = ((fingerprint_input.get("input") or {}).get("input_root"))
    if not stored_input_root_value:
        return str(requested_value)
    stored_input_root = Path(str(stored_input_root_value)).expanduser()
    if stored_input_root.name != "preprocessed":
        return str(requested_value)
    stored_data_root = stored_input_root.parent
    stored_absolute = (
        stored_data_root.resolve()
        if stored_data_root.is_absolute()
        else (REPOSITORY_ROOT / stored_data_root).resolve()
    )
    if stored_absolute != resolved_data_root.resolve():
        return str(requested_value)
    return str(stored_data_root)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(
                f"Manifest header must be exactly {list(MANIFEST_FIELDS)}, "
                f"got {reader.fieldnames}: {path}"
            )
        rows = [dict(row) for row in reader]

    keys: set[tuple[str, int]] = set()
    checkpoint_owners: dict[Path, tuple[str, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset_id = str(row.get("dataset_id") or "").strip()
        checkpoint_value = str(row.get("checkpoint_path") or "").strip()
        pretrained_model = str(row.get("pretrained_model") or "").strip()
        try:
            seed = int(str(row.get("seed") or "").strip())
        except ValueError as exc:
            raise ValueError(f"Invalid seed on manifest row {row_number}: {path}") from exc
        if not dataset_id or not checkpoint_value:
            raise ValueError(f"Manifest row {row_number} contains an empty required value: {path}")
        key = (dataset_id, seed)
        if key in keys:
            raise ValueError(f"Manifest contains duplicate dataset/seed entry {key}: {path}")
        keys.add(key)
        checkpoint_path = _resolve_manifest_checkpoint(path, checkpoint_value)
        previous_owner = checkpoint_owners.get(checkpoint_path)
        if previous_owner is not None and previous_owner != key:
            raise ValueError(
                f"Manifest reuses checkpoint {checkpoint_path} for {previous_owner} and {key}."
            )
        checkpoint_owners[checkpoint_path] = key
        row["dataset_id"] = dataset_id
        row["seed"] = str(seed)
        row["checkpoint_path"] = checkpoint_value
        row["pretrained_model"] = pretrained_model
    return rows


def _append_manifest_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_manifest_row(
    path: Path,
    rows: list[dict[str, str]],
    key: tuple[str, int],
    replacement: dict[str, str],
) -> None:
    updated: list[dict[str, str]] = []
    replaced = False
    for row in rows:
        row_key = (row["dataset_id"], int(row["seed"]))
        if row_key == key:
            updated.append(replacement)
            replaced = True
        else:
            updated.append(row)
    if not replaced:
        raise KeyError(f"Cannot replace absent manifest entry: {key}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
def _seed_run_directories(run_root: Path, dataset_id: str, seed: int) -> set[Path]:
    seed_root = run_root / dataset_id / f"seed_{seed}"
    if not seed_root.is_dir():
        return set()
    return {path.resolve() for path in seed_root.iterdir() if path.is_dir()}


def _read_summary_from_new_run(new_run_dirs: set[Path]) -> tuple[Path, dict[str, Any]]:
    if len(new_run_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one new run directory, found {len(new_run_dirs)}: "
            f"{sorted(str(path) for path in new_run_dirs)}"
        )
    run_dir = next(iter(new_run_dirs))
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed legacy run did not create summary.json: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid summary JSON: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise TypeError(f"Summary root must be an object: {summary_path}")
    return run_dir, summary


def _normalize_active_views(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = []
    return tuple(dict.fromkeys(values))


def _validate_candidate_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_id: str,
    seed: int,
) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    required = {"model_state_dict", "fusion_config", "primary_task_spec"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"Checkpoint lacks required keys {sorted(missing)}: {checkpoint_path}"
        )
    if not isinstance(payload["model_state_dict"], dict) or not payload["model_state_dict"]:
        raise ValueError(f"Checkpoint model_state_dict is empty: {checkpoint_path}")
    if not isinstance(payload["primary_task_spec"], dict):
        raise TypeError(f"Checkpoint primary_task_spec must be a mapping: {checkpoint_path}")
    fusion_config = payload.get("fusion_config")
    if not isinstance(fusion_config, dict):
        raise ValueError(f"Checkpoint lacks fusion_config: {checkpoint_path}")
    if fusion_config.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Checkpoint dataset mismatch: expected {dataset_id}, "
            f"got {fusion_config.get('dataset_id')}: {checkpoint_path}"
        )
    if str(fusion_config.get("mode", "full")).strip().lower() != "full":
        raise ValueError(f"Checkpoint mode must be full: {checkpoint_path}")
    if str(fusion_config.get("fusion_variant", "poe")).strip().lower() != "poe":
        raise ValueError(f"Checkpoint fusion variant must be PoE: {checkpoint_path}")
    checkpoint_seed = payload.get("seed", fusion_config.get("data_loader_seed"))
    if checkpoint_seed is None or int(checkpoint_seed) != int(seed):
        raise ValueError(
            f"Checkpoint seed mismatch: expected {seed}, got {checkpoint_seed}: {checkpoint_path}"
        )
    active_views = _normalize_active_views(
        payload.get("active_views", fusion_config.get("active_views"))
    )
    if len(active_views) != len(STANDARD_ACTIVE_VIEWS) or set(active_views) != set(
        STANDARD_ACTIVE_VIEWS
    ):
        raise ValueError(
            f"Checkpoint must contain exactly C/S/P active views; got {list(active_views)}: "
            f"{checkpoint_path}"
        )
    if payload.get("selection_scope") != BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS:
        raise ValueError(f"Checkpoint is not candidate-epoch selected: {checkpoint_path}")
    if payload.get("selection_rule") != BEST_SELECTION_RULE_RAW_MAX:
        raise ValueError(f"Checkpoint does not use raw candidate monitor maximum: {checkpoint_path}")
    try:
        epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        candidate_start = int(payload["candidate_epoch_start"])
        candidate_end = int(payload["candidate_epoch_end"])
        selection_min_delta = float(payload["best_selection_min_delta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint lacks valid candidate-selection metadata: {checkpoint_path}"
        ) from exc
    if epoch != best_epoch:
        raise ValueError(
            f"Checkpoint epoch {epoch} differs from best_epoch {best_epoch}: {checkpoint_path}"
        )
    if candidate_start < 1 or candidate_end < candidate_start:
        raise ValueError(f"Checkpoint candidate range is invalid: {checkpoint_path}")
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
    if not candidate_start <= epoch <= candidate_end:
        raise ValueError(
            f"Checkpoint epoch {epoch} is outside candidate range "
            f"[{candidate_start}, {candidate_end}]: {checkpoint_path}"
        )
    if selection_min_delta != 0.0:
        raise ValueError(f"Candidate checkpoint selection min_delta must be zero: {checkpoint_path}")
    return {
        "epoch": epoch,
        "candidate_epoch_start": candidate_start,
        "candidate_epoch_end": candidate_end,
    }


def _manifest_row_from_summary(
    *,
    summary: dict[str, Any],
    run_dir: Path,
    dataset_id: str,
    seed: int,
    manifest_path: Path,
    pretrained_model_override: str | None = None,
) -> dict[str, str]:
    run = summary.get("run") or {}
    if run.get("dataset_id") != dataset_id or int(run.get("seed", -1)) != seed:
        raise ValueError(
            f"Summary identity mismatch: expected dataset={dataset_id} seed={seed}, "
            f"got dataset={run.get('dataset_id')} seed={run.get('seed')}"
        )
    testing = summary.get("testing") or {}
    if testing.get("auto_test_ran") is not True:
        raise ValueError(f"Legacy candidate-epoch test did not complete for {dataset_id} seed={seed}.")
    active_views = list((summary.get("data") or {}).get("active_views") or [])
    if len(active_views) != len(STANDARD_ACTIVE_VIEWS) or set(active_views) != set(STANDARD_ACTIVE_VIEWS):
        raise ValueError(
            f"Standard checkpoint requires exactly C/S/P active views; got {active_views} "
            f"for {dataset_id} seed={seed}."
        )

    training = summary.get("training") or {}
    checkpoint_value = training.get("best_checkpoint")
    if not checkpoint_value:
        raise ValueError(f"Legacy run did not report a candidate-best checkpoint for {dataset_id} seed={seed}.")
    checkpoint_path = Path(str(checkpoint_value)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPOSITORY_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.name != "checkpoint_best.pt" or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Candidate-best checkpoint is missing or misnamed: {checkpoint_path}")
    expected_checkpoint = (run_dir / "checkpoint_best.pt").resolve()
    if checkpoint_path != expected_checkpoint:
        raise ValueError(
            f"Summary checkpoint does not belong to the newly completed run: "
            f"expected {expected_checkpoint}, got {checkpoint_path}"
        )
    checkpoint_identity = _validate_candidate_checkpoint(
        checkpoint_path,
        dataset_id=dataset_id,
        seed=seed,
    )
    try:
        training_best_epoch = int(training["best_epoch"])
        testing_best_epoch = int(testing["best_epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Summary lacks valid training/testing best epochs for {dataset_id} seed={seed}."
        ) from exc
    if not (
        checkpoint_identity["epoch"] == training_best_epoch == testing_best_epoch
    ):
        raise ValueError(
            "Candidate-best epoch mismatch among checkpoint, training summary, and testing "
            f"summary for {dataset_id} seed={seed}: checkpoint={checkpoint_identity['epoch']}, "
            f"training={training_best_epoch}, testing={testing_best_epoch}."
        )

    pretrained_models = (summary.get("hyperparameters") or {}).get("pretrained_models") or {}
    configured_models = {
        view_name: (
            pretrained_model_override
            if pretrained_model_override is not None
            else resolve_encoder_config(view_name, dataset_id, "full").model_name("full")
        )
        for view_name in ("content", "stance")
    }
    content_model = str(
        pretrained_models.get("content") or configured_models["content"]
    ).strip()
    stance_model = str(
        pretrained_models.get("stance") or configured_models["stance"]
    ).strip()
    if content_model != stance_model:
        raise ValueError(
            f"Content and stance use different pretrained models: {content_model!r} vs {stance_model!r}."
        )

    relative_checkpoint = os.path.relpath(checkpoint_path, start=manifest_path.parent.resolve())
    return {
        "dataset_id": dataset_id,
        "seed": str(seed),
        "checkpoint_path": Path(relative_checkpoint).as_posix(),
        "pretrained_model": content_model,
    }


def _training_command(
    *,
    dataset_id: str,
    seed: int,
    pretrained_model: str | None,
    device: str,
    data_root: str | Path,
    run_root: str | Path,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_ENTRYPOINT),
        "--dataset_id",
        dataset_id,
        "--seed",
        str(seed),
        "--checkpoint",
        "true",
        "--device",
        device,
        "--data_root",
        str(data_root),
        "--run_root",
        str(run_root),
    ]
    if pretrained_model is not None:
        command.extend(["--pretrained_model", pretrained_model])
    return command


def run_seed_sweep(args: argparse.Namespace) -> None:
    dataset_id = str(args.dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty.")
    pretrained_model = None
    if args.pretrained_model is not None:
        pretrained_model = str(args.pretrained_model).strip()
        if not pretrained_model:
            raise ValueError("pretrained_model must not be empty.")

    # Artifact fingerprints include the input-root string. Preserve the path
    # spelling already stored in metadata so relative and absolute spellings of
    # the same directory do not invalidate one another's caches.
    resolved_data_root = _absolute_from_repository_root(args.data_root)
    data_root_argument = _artifact_compatible_data_root_argument(
        requested_value=args.data_root,
        resolved_data_root=resolved_data_root,
        dataset_id=dataset_id,
    )
    run_root_argument = str(args.run_root)
    run_root = _absolute_from_repository_root(args.run_root)
    manifest_path = _absolute_from_repository_root(args.manifest)
    rows = _load_manifest(manifest_path)
    rows_by_key = {(row["dataset_id"], int(row["seed"])): row for row in rows}

    for seed in SEEDS:
        key = (dataset_id, seed)
        existing = rows_by_key.get(key)
        existing_checkpoint = (
            _resolve_manifest_checkpoint(manifest_path, existing["checkpoint_path"])
            if existing is not None
            else None
        )
        if existing_checkpoint is not None and existing_checkpoint.is_file():
            try:
                _validate_candidate_checkpoint(
                    existing_checkpoint,
                    dataset_id=dataset_id,
                    seed=seed,
                )
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                print(
                    f"Rebuilding stale standard checkpoint: dataset={dataset_id} seed={seed} "
                    f"checkpoint={existing_checkpoint} reason={exc}"
                )
            else:
                print(
                    f"Skipping existing candidate-best checkpoint: dataset={dataset_id} "
                    f"seed={seed} checkpoint={existing_checkpoint}"
                )
                continue

        before = _seed_run_directories(run_root, dataset_id, seed)
        command = _training_command(
            dataset_id=dataset_id,
            seed=seed,
            pretrained_model=pretrained_model,
            device=args.device,
            data_root=data_root_argument,
            run_root=run_root_argument,
        )
        print(f"Running legacy UMCL pipeline: dataset={dataset_id} seed={seed}")
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        after = _seed_run_directories(run_root, dataset_id, seed)
        run_dir, summary = _read_summary_from_new_run(after - before)
        manifest_row = _manifest_row_from_summary(
            summary=summary,
            run_dir=run_dir,
            dataset_id=dataset_id,
            seed=seed,
            manifest_path=manifest_path,
            pretrained_model_override=pretrained_model,
        )

        if existing is None:
            _append_manifest_row(manifest_path, manifest_row)
            rows.append(manifest_row)
        else:
            _replace_manifest_row(manifest_path, rows, key, manifest_row)
            rows = [
                manifest_row
                if (row["dataset_id"], int(row["seed"])) == key
                else row
                for row in rows
            ]
        rows_by_key[key] = manifest_row
        print(
            f"Registered standard checkpoint: dataset={dataset_id} seed={seed} "
            f"checkpoint={manifest_row['checkpoint_path']}"
        )

def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_seed_sweep(parse_args(argv))
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
