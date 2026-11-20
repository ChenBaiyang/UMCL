"""UMCL experiment: fusion efficiency."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from config.resolver import resolve_fusion_config
from models.artifacts import EnsureArtifactsOptions, ensure_model_inputs
from models.fusion.factory import create_fusion_model
from models.fusion.internal_fusion_model import _masked_mean
from models.fusion.posteriors import product_of_experts, reparameterize
from models.fusion.schemas import ModalityPosterior
from models.fusion.view_mask import VIEW_ORDER, ViewMask
from models.evaluation import build_test_loader_with_fallback
from models.fusion.evaluation import _move_batch_to_device
from utils.torch_runtime import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent
VARIANT_SPECS = (
    ("concat", "MLP Posterior Fusion"),
    ("gated", "Gated Posterior Fusion"),
    ("mean_posterior", "Mean Posterior Fusion"),
    ("poe", "PoE Fusion"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled post-encoder efficiency comparison for UMCL fusion strategies."
    )
    parser.add_argument("--dataset_id", default="PHEME")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--active_views", default="content,propagation,stance")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="results/computational_efficiency/fusion/pheme",
    )
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser.parse_args()


def _parse_views(raw: str) -> list[str]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(VIEW_ORDER))
    if invalid:
        raise ValueError(f"Unknown active views: {invalid}")
    views = [view for view in VIEW_ORDER if view in requested]
    if len(views) != 3 or len(set(requested)) != 3:
        raise ValueError("--active_views must contain exactly three distinct views.")
    return views


def _resolve_device(choice: str, cuda_index: int) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal benchmark; use --device cpu only for debugging.")
    if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device index {cuda_index}; count={torch.cuda.device_count()}")
    return torch.device(f"cuda:{cuda_index}")


def _output_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _posterior_head(model: torch.nn.Module, view: str) -> torch.nn.Module:
    encoder = getattr(model, f"{view}_encoder")
    if encoder is None:
        raise ValueError(f"Model does not contain the requested {view!r} encoder.")
    if hasattr(encoder, "posterior_heads"):
        return encoder.posterior_heads
    if hasattr(encoder, "posterior"):
        return encoder.posterior
    raise AttributeError(f"Could not locate posterior heads for view={view!r}")


def _encode_view_hidden(model: torch.nn.Module, batch: Any, view: str) -> ModalityPosterior:
    if view == "content":
        posterior, _ = model._encode_content(batch)
        return posterior
    if view == "stance":
        posterior, _, _ = model._encode_stance(batch)
        return posterior
    if view == "propagation":
        posterior, _ = model._encode_propagation(batch)
        return posterior
    if view == "evidence":
        posterior, _ = model._encode_evidence(batch)
        return posterior
    if view == "source":
        posterior, _ = model._encode_source(batch)
        return posterior
    raise ValueError(f"Unsupported view: {view}")


def _cache_matches(
    payload: dict[str, Any],
    *,
    dataset_id: str,
    active_views: list[str],
    seed: int,
    max_batches: int | None,
) -> bool:
    metadata = payload.get("metadata") or {}
    hidden = payload.get("hidden") or {}
    masks = payload.get("masks") or {}
    return (
        metadata.get("dataset_id") == dataset_id
        and metadata.get("active_views") == active_views
        and metadata.get("seed") == seed
        and metadata.get("max_batches") == max_batches
        and all(view in hidden and view in masks for view in active_views)
    )


def _build_representation_cache(
    *,
    dataset_id: str,
    data_root: Path,
    active_views: list[str],
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int | None,
    max_batches: int | None,
) -> dict[str, Any]:
    config = resolve_fusion_config(dataset_id=dataset_id, mode="full")
    config.fusion_variant = "poe"
    config.data_loader_seed = seed
    ensure_model_inputs(
        dataset_id=dataset_id,
        mode="full",
        active_views=active_views,
        config=config,
        data_root=data_root,
        input_root=data_root / "preprocessed",
        options=EnsureArtifactsOptions(),
    )
    (
        loader,
        _sample_count,
        selected_splits,
        feature_dims,
        loader_settings,
        primary_task_spec,
    ) = build_test_loader_with_fallback(
        dataset_id=dataset_id,
        config=config,
        data_root=data_root,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=None,
        disable_persistent_workers=False,
        disable_pin_memory=False,
        required_views=set(active_views),
    )
    model = create_fusion_model(
        config,
        feature_dims,
        primary_task_spec=primary_task_spec,
        active_views=set(active_views),
    ).to(device)
    model.eval()
    hidden_parts: dict[str, list[torch.Tensor]] = {view: [] for view in active_views}
    mask_parts: dict[str, list[torch.Tensor]] = {view: [] for view in active_views}
    event_ids: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            _move_batch_to_device(batch, device, non_blocking=device.type == "cuda")
            event_ids.extend(str(event_id) for event_id in batch.event_ids)
            for view in active_views:
                posterior = _encode_view_hidden(model, batch, view)
                hidden_parts[view].append(posterior.hidden.detach().float().cpu())
                mask_parts[view].append(posterior.mask.detach().float().cpu())
    if not event_ids:
        raise ValueError("The representation loader produced zero events.")
    payload = {
        "metadata": {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "active_views": active_views,
            "seed": seed,
            "max_batches": max_batches,
            "sample_count": len(event_ids),
            "selected_splits": selected_splits,
            "feature_dims": feature_dims,
            "primary_task_spec": primary_task_spec,
            "loader_settings": loader_settings,
            "weights": "random_initialization",
        },
        "event_ids": event_ids,
        "hidden": {view: torch.cat(hidden_parts[view], dim=0) for view in active_views},
        "masks": {view: torch.cat(mask_parts[view], dim=0) for view in active_views},
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _load_or_build_cache(
    *,
    cache_path: Path,
    args: argparse.Namespace,
    dataset_id: str,
    data_root: Path,
    active_views: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and _cache_matches(
            payload,
            dataset_id=dataset_id,
            active_views=active_views,
            seed=args.seed,
            max_batches=args.max_batches,
        ):
            return payload, True
    payload = _build_representation_cache(
        dataset_id=dataset_id,
        data_root=data_root,
        active_views=active_views,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload, False


def _unique_parameter_count(modules: Sequence[torch.nn.Module]) -> int:
    seen: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen or not parameter.requires_grad:
                continue
            seen.add(id(parameter))
            count += parameter.numel()
    return count


def _move_modules(modules: Sequence[torch.nn.Module], device: torch.device) -> None:
    seen: set[int] = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        module.to(device=device, dtype=torch.float32)
        module.eval()


def _representation_batches(
    payload: dict[str, Any], active_views: list[str], batch_size: int, device: torch.device,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    sample_count = int(payload["metadata"]["sample_count"])
    batches: list[dict[str, dict[str, torch.Tensor]]] = []
    for start in range(0, sample_count, batch_size):
        stop = min(sample_count, start + batch_size)
        batches.append({
            view: {
                "hidden": payload["hidden"][view][start:stop].to(device=device, dtype=torch.float32),
                "mask": payload["masks"][view][start:stop].to(device=device, dtype=torch.float32),
            }
            for view in active_views
        })
    return batches


def _projection_forward(
    model: torch.nn.Module,
    batch: dict[str, dict[str, torch.Tensor]],
    subset: tuple[str, ...],
    active_views: list[str],
) -> torch.Tensor:
    subset_set = set(subset)
    modality_posteriors: dict[str, ModalityPosterior] = {}
    for view in active_views:
        hidden = batch[view]["hidden"]
        natural_mask = batch[view]["mask"]
        mask = natural_mask if view in subset_set else torch.zeros_like(natural_mask)
        masked_hidden = hidden if view in subset_set else torch.zeros_like(hidden)
        latent_zeros = hidden.new_zeros((hidden.shape[0], model.config.posterior_latent_dim))
        modality_posteriors[view] = ModalityPosterior(
            hidden=masked_hidden,
            mu=latent_zeros,
            logvar=latent_zeros,
            mask=mask,
        )
    output = model._forward_concat_or_gated(
        modality_posteriors,
        reply_logits=None,
        view_mask=ViewMask(active_views=set(subset)),
    )
    return torch.softmax(output.task_outputs["primary_task"]["logits"], dim=-1)


def _posterior_forward(
    *,
    variant: str,
    model: torch.nn.Module,
    posterior_heads: dict[str, torch.nn.Module],
    batch: dict[str, dict[str, torch.Tensor]],
    subset: tuple[str, ...],
) -> torch.Tensor:
    mus: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for view in subset:
        mu, logvar = posterior_heads[view](batch[view]["hidden"])
        mus.append(mu)
        logvars.append(logvar)
        masks.append(batch[view]["mask"])
    if variant == "mean_posterior":
        mask_stack = torch.stack(masks, dim=1)
        posterior_mu = _masked_mean(torch.stack(mus, dim=1), mask_stack)
        posterior_logvar = torch.clamp(
            _masked_mean(torch.stack(logvars, dim=1), mask_stack),
            min=model.config.posterior_logvar_clip_min,
            max=model.config.posterior_logvar_clip_max,
        )
    else:
        posterior_mu, posterior_logvar = product_of_experts(
            mus=mus,
            logvars=logvars,
            masks=masks,
            use_unit_prior_expert=model.config.use_unit_prior_expert,
            logvar_min=model.config.posterior_logvar_clip_min,
            logvar_max=model.config.posterior_logvar_clip_max,
        )
    z = reparameterize(
        posterior_mu,
        posterior_logvar,
        training=False,
        sample_during_training=False,
    )
    logits = model.task_heads["primary_task"](z)
    return torch.softmax(logits, dim=-1)


def _timed_pass(
    *,
    forward_fn: Any,
    batches: Sequence[dict[str, dict[str, torch.Tensor]]],
    device: torch.device,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for batch in batches:
            forward_fn(batch)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
    else:
        started = time.perf_counter()
        for batch in batches:
            forward_fn(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms / len(batches)


def _benchmark_variant(
    *,
    variant: str,
    display_name: str,
    base_config: Any,
    feature_dims: dict[str, int],
    primary_task_spec: dict[str, Any],
    active_views: list[str],
    batches: Sequence[dict[str, dict[str, torch.Tensor]]],
    device: torch.device,
    warmup_runs: int,
    benchmark_runs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = copy.deepcopy(base_config)
    config.fusion_variant = variant
    model = create_fusion_model(
        config,
        feature_dims,
        primary_task_spec=primary_task_spec,
        active_views=set(active_views),
    )
    if variant in {"concat", "gated"}:
        counted_modules: list[torch.nn.Module] = [model.view_projection_layers, model.task_heads["primary_task"]]
        if variant == "gated":
            counted_modules.append(model.view_gate_layers)
        _move_modules(counted_modules, device)

        def make_forward(subset: tuple[str, ...]):
            return lambda batch: _projection_forward(model, batch, subset, active_views)

    else:
        posterior_heads = {view: _posterior_head(model, view) for view in active_views}
        counted_modules = [*posterior_heads.values(), model.task_heads["primary_task"]]
        _move_modules(counted_modules, device)

        def make_forward(subset: tuple[str, ...]):
            return lambda batch: _posterior_forward(
                variant=variant,
                model=model,
                posterior_heads=posterior_heads,
                batch=batch,
                subset=subset,
            )

    trainable_parameters = _unique_parameter_count(counted_modules)
    subset_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for view_count in (1, 2, 3):
            for subset in itertools.combinations(active_views, view_count):
                forward_fn = make_forward(subset)
                for _ in range(warmup_runs):
                    _timed_pass(forward_fn=forward_fn, batches=batches, device=device)
                repeats = [
                    _timed_pass(forward_fn=forward_fn, batches=batches, device=device)
                    for _ in range(benchmark_runs)
                ]
                subset_rows.append({
                    "variant": variant,
                    "fusion_strategy": display_name,
                    "view_count": view_count,
                    "view_subset": "+".join(subset),
                    "trainable_parameters": trainable_parameters,
                    "batch_count": len(batches),
                    "latency_ms_per_batch_mean": statistics.fmean(repeats),
                    "latency_ms_per_batch_std": statistics.stdev(repeats) if len(repeats) > 1 else 0.0,
                    "repeat_latency_ms_per_batch": repeats,
                })
    summary: dict[str, Any] = {
        "variant": variant,
        "fusion_strategy": display_name,
        "trainable_parameters": trainable_parameters,
    }
    for view_count in (1, 2, 3):
        values = [
            latency
            for row in subset_rows
            if row["view_count"] == view_count
            for latency in row["repeat_latency_ms_per_batch"]
        ]
        summary[f"latency_{view_count}_view_ms_per_batch_mean"] = statistics.fmean(values)
        summary[f"latency_{view_count}_view_ms_per_batch_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    for module in counted_modules:
        module.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, subset_rows


def _write_outputs(
    *,
    output_dir: Path,
    raw_payload: dict[str, Any],
    summaries: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fusion_efficiency_raw.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subset_fields = [
        "variant", "fusion_strategy", "view_count", "view_subset", "trainable_parameters",
        "batch_count", "latency_ms_per_batch_mean", "latency_ms_per_batch_std",
        "repeat_latency_ms_per_batch",
    ]
    with (output_dir / "fusion_efficiency_per_subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=subset_fields)
        writer.writeheader()
        for row in subset_rows:
            csv_row = dict(row)
            csv_row["repeat_latency_ms_per_batch"] = json.dumps(row["repeat_latency_ms_per_batch"])
            writer.writerow(csv_row)
    summary_fields = [
        "variant", "fusion_strategy", "trainable_parameters",
        "latency_1_view_ms_per_batch_mean", "latency_1_view_ms_per_batch_std",
        "latency_2_view_ms_per_batch_mean", "latency_2_view_ms_per_batch_std",
        "latency_3_view_ms_per_batch_mean", "latency_3_view_ms_per_batch_std",
    ]
    with (output_dir / "fusion_efficiency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    latex_lines: list[str] = []
    for row in summaries:
        label = r"\textbf{PoE Fusion}" if row["variant"] == "poe" else row["fusion_strategy"]
        latex_lines.append(
            f'{label} & {row["trainable_parameters"]} & '
            f'{row["latency_1_view_ms_per_batch_mean"]:.4f} & '
            f'{row["latency_2_view_ms_per_batch_mean"]:.4f} & '
            f'{row["latency_3_view_ms_per_batch_mean"]:.4f} \\\\'
        )
    (output_dir / "fusion_efficiency_rows.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.warmup_runs < 0 or args.benchmark_runs <= 0:
        raise ValueError("--warmup_runs must be >= 0 and --benchmark_runs must be > 0.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max_batches must be positive when provided.")
    dataset_id = args.dataset_id.strip()
    active_views = _parse_views(args.active_views)
    device = _resolve_device(args.device, args.cuda_index)
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    output_dir = _output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "precomputed_event_representations.pt"
    seed_everything(args.seed)
    cache, cache_reused = _load_or_build_cache(
        cache_path=cache_path,
        args=args,
        dataset_id=dataset_id,
        data_root=data_root,
        active_views=active_views,
        device=device,
    )
    batches = _representation_batches(cache, active_views, args.batch_size, device)
    if not batches:
        raise ValueError("Representation cache produced zero batches.")
    base_config = resolve_fusion_config(dataset_id=dataset_id, mode="full")
    base_config.data_loader_seed = args.seed
    feature_dims = dict(cache["metadata"]["feature_dims"])
    primary_task_spec = dict(cache["metadata"]["primary_task_spec"])
    summaries: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    for variant, display_name in VARIANT_SPECS:
        summary, rows = _benchmark_variant(
            variant=variant,
            display_name=display_name,
            base_config=base_config,
            feature_dims=feature_dims,
            primary_task_spec=primary_task_spec,
            active_views=active_views,
            batches=batches,
            device=device,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        summaries.append(summary)
        subset_rows.extend(rows)
        print(
            f'{display_name}: params={summary["trainable_parameters"]} '
            f'1-view={summary["latency_1_view_ms_per_batch_mean"]:.4f} ms/batch '
            f'2-view={summary["latency_2_view_ms_per_batch_mean"]:.4f} ms/batch '
            f'3-view={summary["latency_3_view_ms_per_batch_mean"]:.4f} ms/batch',
            flush=True,
        )
    availability = {
        view: {
            "available": int((cache["masks"][view] > 0).sum().item()),
            "missing": int((cache["masks"][view] <= 0).sum().item()),
        }
        for view in active_views
    }
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(args.cuda_index) if device.type == "cuda" else None,
    }
    raw_payload = {
        "schema_version": 1,
        "experiment": "architecture_only_post_encoder_efficiency",
        "dataset_id": dataset_id,
        "active_views": active_views,
        "weights": "random_initialization",
        "precision": "fp32",
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "sample_count": int(cache["metadata"]["sample_count"]),
        "batch_count": len(batches),
        "availability": availability,
        "cache_path": str(cache_path),
        "cache_reused": cache_reused,
        "summaries": summaries,
        "subsets": subset_rows,
    }
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "command": sys.argv,
        "arguments": vars(args),
        "environment": environment,
        "representation_cache": cache["metadata"],
        "outputs": {
            "raw_json": str(output_dir / "fusion_efficiency_raw.json"),
            "per_subset_csv": str(output_dir / "fusion_efficiency_per_subset.csv"),
            "summary_csv": str(output_dir / "fusion_efficiency_summary.csv"),
            "latex": str(output_dir / "fusion_efficiency_rows.tex"),
        },
    }
    _write_outputs(
        output_dir=output_dir,
        raw_payload=raw_payload,
        summaries=summaries,
        subset_rows=subset_rows,
        metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
