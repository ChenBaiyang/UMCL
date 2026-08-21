from __future__ import annotations

import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from config.resolver import resolve_encoder_config
from models.fusion.collator import FusionBatchCollator, collate_fusion_batch
from models.fusion.config import FusionConfig, validate_monitor_name
from models.fusion.dataset import FiveViewDataset
from models.fusion.evaluation import run_evaluation
from models.fusion.losses import build_loss_weights, compute_total_loss
from models.fusion.model import UMCLFiveViewModel
from utils.logging_utils import get_logger
from utils.torch_runtime import seed_worker

if TYPE_CHECKING:
    from models.fusion.subset_sampling import DeterministicMixedSubsetMasker
    from models.fusion.view_mask import ViewMask


LOGGER = get_logger("fusion_training")


@dataclass
class TrainStepRecord:
    step: int
    epoch: int
    batch_size: int
    dataset_id: str
    active_views: list[str]
    total_loss: float
    reconstruction_loss: float
    task_primary_loss: float
    aux_stance_loss: float
    kl_loss: float
    reconstruction_losses: dict[str, float] = field(default_factory=dict)


@dataclass
class EpochRecord:
    epoch: int
    train_example_count: int
    eval_example_count: int | None
    active_views: list[str]
    train_losses: dict[str, float]
    eval_losses: dict[str, float] | None
    monitor_name: str | None
    monitor_value: float | None
    head_lr: float
    optimizer_group_lrs: dict[str, float]
    is_best: bool
    best_epoch: int | None
    best_monitor_value: float | None
    early_stop_counter: int


def build_eval_monitor_name(primary_task_spec: dict[str, Any], monitor_name: str) -> str:
    return f"eval_{primary_task_spec['metric_name']}_{validate_monitor_name(monitor_name)}"


def compute_eval_monitor_value(primary_stats: dict[str, Any], monitor_name: str) -> float:
    normalized_monitor_name = validate_monitor_name(monitor_name)
    if normalized_monitor_name == "macro_f1":
        return float(primary_stats["macro_f1"])
    if normalized_monitor_name == "accuracy":
        return float(primary_stats["accuracy"])
    if normalized_monitor_name == "composite":
        return float(0.7 * float(primary_stats["macro_f1"]) + 0.3 * float(primary_stats["accuracy"]))
    raise ValueError(f"Unsupported monitor_name: {normalized_monitor_name}")


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        subset: Subset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.subset = subset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.min_last_batch_size = max(1, self.batch_size // 4)
        self._epoch = 0
        self._bucket_map = self._build_bucket_map()

    def _build_bucket_map(self) -> dict[int, list[int]]:
        dataset = self.subset.dataset
        if not isinstance(dataset, FiveViewDataset):
            raise TypeError("BucketBatchSampler requires a FiveViewDataset-backed subset.")
        buckets: dict[int, list[int]] = {}
        for subset_pos, dataset_index in enumerate(self.subset.indices):
            bucket_id = dataset.bucket_id_at(int(dataset_index))
            buckets.setdefault(bucket_id, []).append(subset_pos)
        return buckets

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        bucket_lists: list[list[int]] = []
        for bucket_id in sorted(self._bucket_map):
            indices = self._bucket_map[bucket_id]
            current = list(indices)
            if self.shuffle:
                rng.shuffle(current)
            if current:
                bucket_lists.append(current)
        bucket_positions = [0] * len(bucket_lists)
        active_bucket_ids = list(range(len(bucket_lists)))
        cursor = 0
        batch: list[int] = []
        while active_bucket_ids:
            if cursor >= len(active_bucket_ids):
                cursor = 0
            active_id = active_bucket_ids[cursor]
            position = bucket_positions[active_id]
            current = bucket_lists[active_id]
            batch.append(current[position])
            position += 1
            bucket_positions[active_id] = position
            if position >= len(current):
                active_bucket_ids.pop(cursor)
                if not active_bucket_ids:
                    cursor = 0
                elif cursor >= len(active_bucket_ids):
                    cursor = 0
            else:
                cursor = (cursor + 1) % len(active_bucket_ids)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last and len(batch) >= self.min_last_batch_size:
            yield batch

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self._bucket_map.values())
        if self.drop_last:
            return total // self.batch_size
        full_batches, remainder = divmod(total, self.batch_size)
        if remainder == 0:
            return full_batches
        return full_batches + (1 if remainder >= self.min_last_batch_size else 0)


def _stringify_split(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return "unknown"
    return str(value)


def infer_default_splits(dataset: FiveViewDataset) -> list[str]:
    splits = [_stringify_split(v) for v in dataset.event_index["canonical_splits"]]
    unique = {s for s in splits}
    if "train" in unique:
        return ["train"]
    if "all" in unique:
        return ["all"]
    return sorted(unique)


def infer_default_eval_splits(dataset: FiveViewDataset) -> list[str]:
    splits = [_stringify_split(v) for v in dataset.event_index["canonical_splits"]]
    unique = {s for s in splits}
    if "val" in unique:
        return ["val"]
    if "dev" in unique:
        return ["dev"]
    if "test" in unique:
        return ["test"]
    if "all" in unique:
        return ["all"]
    if "train" in unique:
        return ["train"]
    return sorted(unique)


def _loader_settings(config: FusionConfig, device: torch.device) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    num_workers = min(config.max_auto_workers, max(1, cpu_count - 1))
    return {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": device.type == "cuda",
    }


def resolve_loader_settings(
    config: FusionConfig,
    device: torch.device,
    *,
    num_workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    persistent_workers_override: bool | None = None,
    pin_memory_override: bool | None = None,
) -> dict[str, Any]:
    settings = _loader_settings(config, device)
    if num_workers_override is not None:
        settings["num_workers"] = max(0, int(num_workers_override))
    if pin_memory_override is not None:
        settings["pin_memory"] = bool(pin_memory_override)
    if settings["num_workers"] > 0:
        if persistent_workers_override is not None:
            settings["persistent_workers"] = bool(persistent_workers_override)
        else:
            settings["persistent_workers"] = True
        if prefetch_factor_override is not None:
            settings["prefetch_factor"] = max(1, int(prefetch_factor_override))
    else:
        settings["persistent_workers"] = False
    return settings


def _loader_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _loader_worker_init_fn(seed: int | None):
    if seed is None:
        return None
    return partial(seed_worker, seed=int(seed))


def build_subset(
    dataset_id: str,
    config: FusionConfig,
    data_root: Path,
    splits: list[str] | None,
    *,
    required_views: set[str] | None = None,
    load_bucket_metadata: bool = False,
) -> Dataset:
    dataset = FiveViewDataset(
        dataset_id=dataset_id,
        mode=config.mode,
        data_root=data_root,
        cache_max_open_shards=config.max_open_shards_limit(),
        required_views=required_views,
        load_bucket_metadata=load_bucket_metadata,
    )
    selected_splits = splits or infer_default_splits(dataset)
    split_values = [_stringify_split(v) for v in dataset.event_index["canonical_splits"]]
    indices = [idx for idx, split in enumerate(split_values) if split in selected_splits]
    if not indices:
        raise ValueError(f"No samples found for dataset={dataset_id} splits={selected_splits}")
    subset = Subset(dataset, indices)
    subset.dataset_id = dataset_id  # type: ignore[attr-defined]
    subset.selected_splits = selected_splits  # type: ignore[attr-defined]
    return subset


def build_train_loader(
    dataset_id: str,
    config: FusionConfig,
    data_root: Path,
    batch_size: int,
    splits: list[str] | None,
    device: torch.device,
    seed: int | None = None,
    shuffle: bool = True,
    num_workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    persistent_workers_override: bool | None = None,
    pin_memory_override: bool | None = None,
    required_views: set[str] | None = None,
) -> tuple[DataLoader, int, list[str], dict[str, int], dict[str, Any], dict[str, Any], dict[int, int]]:
    subset = build_subset(
        dataset_id,
        config,
        data_root,
        splits,
        required_views=required_views,
        load_bucket_metadata=config.bucket_batching_enabled(),
    )
    feature_dims = subset.dataset.feature_dims  # type: ignore[attr-defined]
    stance_config = resolve_encoder_config("stance", dataset_id, config.mode)
    propagation_config = resolve_encoder_config("propagation", dataset_id, config.mode)
    loader_kwargs = resolve_loader_settings(
        config,
        device,
        num_workers_override=num_workers_override,
        prefetch_factor_override=prefetch_factor_override,
        persistent_workers_override=persistent_workers_override,
        pin_memory_override=pin_memory_override,
    )
    collate_fn = FusionBatchCollator(
        max_replies_per_event=stance_config.max_replies_per_event_value(),
        max_propagation_nodes_per_event=propagation_config.max_nodes_per_event_value(),
        stochastic_propagation_node_sampling=True,
        propagation_early_fraction=propagation_config.early_fraction,
        propagation_middle_fraction=propagation_config.middle_fraction,
        propagation_sampling_seed=(config.data_loader_seed if seed is None else int(seed)),
    )
    if config.bucket_batching_enabled():
        batch_sampler = BucketBatchSampler(
            subset=subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            seed=config.data_loader_seed if seed is None else int(seed),
        )
        loader = DataLoader(
            subset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=_loader_worker_init_fn(seed),
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            generator=_loader_generator(seed),
            worker_init_fn=_loader_worker_init_fn(seed),
            **loader_kwargs,
        )
    label_counts: dict[int, int] = {}
    dataset = subset.dataset
    if isinstance(dataset, FiveViewDataset):
        for dataset_index in subset.indices:
            label = dataset.primary_label_at(int(dataset_index))
            if label >= 0:
                label_counts[label] = label_counts.get(label, 0) + 1
    return loader, len(subset), subset.selected_splits, feature_dims, loader_kwargs, subset.dataset.primary_task_spec, label_counts  # type: ignore[attr-defined]


def build_eval_loader(
    dataset_id: str,
    config: FusionConfig,
    data_root: Path,
    batch_size: int,
    splits: list[str] | None,
    device: torch.device,
    seed: int | None = None,
    num_workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    persistent_workers_override: bool | None = None,
    pin_memory_override: bool | None = None,
    required_views: set[str] | None = None,
) -> tuple[DataLoader, int, list[str], dict[str, int], dict[str, Any], dict[str, Any]]:
    dataset = FiveViewDataset(
        dataset_id=dataset_id,
        mode=config.mode,
        data_root=data_root,
        cache_max_open_shards=config.max_open_shards_limit(),
        required_views=required_views,
    )
    selected_splits = splits or infer_default_eval_splits(dataset)
    stance_config = resolve_encoder_config("stance", dataset_id, config.mode)
    propagation_config = resolve_encoder_config("propagation", dataset_id, config.mode)
    split_values = [_stringify_split(v) for v in dataset.event_index["canonical_splits"]]
    indices = [idx for idx, split in enumerate(split_values) if split in selected_splits]
    if not indices:
        raise ValueError(f"No evaluation samples found for dataset={dataset_id} splits={selected_splits}")
    subset = Subset(dataset, indices)
    loader_kwargs = resolve_loader_settings(
        config,
        device,
        num_workers_override=num_workers_override,
        prefetch_factor_override=prefetch_factor_override,
        persistent_workers_override=persistent_workers_override,
        pin_memory_override=pin_memory_override,
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fusion_batch,
            max_replies_per_event=stance_config.max_replies_per_event_value(),
            max_propagation_nodes_per_event=propagation_config.max_nodes_per_event_value(),
            stochastic_propagation_node_sampling=False,
            propagation_early_fraction=propagation_config.early_fraction,
            propagation_middle_fraction=propagation_config.middle_fraction,
            propagation_sampling_seed=(config.data_loader_seed if seed is None else int(seed)),
        ),
        generator=_loader_generator(seed),
        worker_init_fn=_loader_worker_init_fn(seed),
        **loader_kwargs,
    )
    return loader, len(subset), selected_splits, dataset.feature_dims, loader_kwargs, dataset.primary_task_spec


def _move_batch_to_device(batch, device: torch.device, *, non_blocking: bool = False) -> None:
    for modality_payload in batch.modalities.values():
        for key, value in modality_payload.items():
            if isinstance(value, torch.Tensor):
                modality_payload[key] = value.to(device, non_blocking=non_blocking)
    for key, value in batch.modality_mask.items():
        batch.modality_mask[key] = value.to(device, non_blocking=non_blocking)
    for key, value in batch.task_labels.items():
        batch.task_labels[key] = value.to(device, non_blocking=non_blocking)
    for key, value in batch.task_mask.items():
        batch.task_mask[key] = value.to(device, non_blocking=non_blocking)


def _autocast_context(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _build_grad_scaler(device: torch.device, use_amp: bool):
    return torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))


def _loss_weights(config: FusionConfig) -> dict[str, float]:
    return build_loss_weights(config)


def _loss_snapshot(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, value in losses.items():
        if key.startswith("_"):
            continue
        snapshot[key] = float(value.detach().cpu().item())
    return snapshot


def _all_losses_finite(losses: dict[str, torch.Tensor]) -> bool:
    for key, value in losses.items():
        if key.startswith("_"):
            continue
        if not torch.isfinite(value).all():
            return False
    return True


def _empty_loss_totals() -> dict[str, float]:
    return {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "task_primary_loss": 0.0,
        "aux_stance_loss": 0.0,
        "kl_loss": 0.0,
    }


def _active_recon_keys(view_mask: "ViewMask | None") -> list[str]:
    if view_mask is None:
        return [f"reconstruction_{view}_loss" for view in ["content", "stance", "propagation", "evidence", "source"]]
    return [f"reconstruction_{view}_loss" for view in view_mask.get_active_views()]


def _loss_keys(view_mask: "ViewMask | None") -> list[str]:
    return [
        "total_loss",
        "reconstruction_loss",
        *_active_recon_keys(view_mask),
        "task_primary_loss",
        "aux_stance_loss",
        "kl_loss",
    ]


def _empty_loss_totals_for_view_mask(view_mask: "ViewMask | None") -> dict[str, float]:
    return _empty_loss_totals()


def _reconstruction_loss_snapshot(losses: dict[str, torch.Tensor], view_mask: "ViewMask | None") -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key in _active_recon_keys(view_mask):
        if key in losses:
            view_name = key[len("reconstruction_"):-len("_loss")]
            snapshot[view_name] = float(losses[key].detach().cpu().item())
    return snapshot


def _format_recon_log(reconstruction_losses: dict[str, float]) -> str:
    if not reconstruction_losses:
        return "none"
    return ", ".join(f"{view}={value:.6f}" for view, value in reconstruction_losses.items())


def _average_loss_totals(sum_losses: dict[str, float], example_count: int) -> dict[str, float]:
    if example_count <= 0:
        raise ValueError("Cannot average losses with zero examples.")
    return {name: value / example_count for name, value in sum_losses.items()}


def save_checkpoint(
    path: Path,
    model: UMCLFiveViewModel,
    config: FusionConfig,
    primary_task_spec: dict[str, Any],
    step: int,
    epoch: int,
    extra_state: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "fusion_config": asdict(config),
        "primary_task_spec": primary_task_spec,
        "step": step,
        "epoch": epoch,
    }
    # if optimizer is not None:
    #     payload["optimizer_state_dict"] = optimizer.state_dict()
    # if scheduler is not None:
    #     payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path, _use_new_zipfile_serialization=False)


def write_train_log(path: Path, records: Iterable[TrainStepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


def write_epoch_log(path: Path, records: Iterable[EpochRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


def _stringify_active_views(active_views: list[str]) -> str:
    return "|".join(active_views)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _epoch_record_to_csv_row(
    record: EpochRecord,
    *,
    dataset_id: str,
    seed: int | None,
    loss_weights: dict[str, float] | None,
) -> dict[str, str]:
    return {
        "dataset_id": _csv_cell(dataset_id),
        "seed": _csv_cell(seed),
        "epoch": _csv_cell(record.epoch),
        "view_reconstruction_loss_weight": _csv_cell(None if loss_weights is None else loss_weights.get("view_reconstruction_loss_weight")),
        "primary_task_loss_weight": _csv_cell(None if loss_weights is None else loss_weights.get("primary_task_loss_weight")),
        "stance_auxiliary_loss_weight": _csv_cell(None if loss_weights is None else loss_weights.get("stance_auxiliary_loss_weight")),
        "posterior_kl_loss_weight": _csv_cell(None if loss_weights is None else loss_weights.get("posterior_kl_loss_weight")),
        "loss_weights": _csv_cell(loss_weights),
        "active_views": _csv_cell(_stringify_active_views(record.active_views)),
        "train_example_count": _csv_cell(record.train_example_count),
        "eval_example_count": _csv_cell(record.eval_example_count),
        "monitor_name": _csv_cell(record.monitor_name),
        "monitor_value": _csv_cell(record.monitor_value),
        "head_lr": _csv_cell(record.head_lr),
        "is_best": _csv_cell(record.is_best),
        "best_epoch": _csv_cell(record.best_epoch),
        "best_monitor_value": _csv_cell(record.best_monitor_value),
        "early_stop_counter": _csv_cell(record.early_stop_counter),
    }


def write_eval_log_csv(
    path: Path,
    records: Iterable[EpochRecord],
    *,
    dataset_id: str,
    seed: int | None,
    loss_weights: dict[str, float] | None,
) -> None:
    fieldnames = [
        "dataset_id",
        "seed",
        "epoch",
        "view_reconstruction_loss_weight",
        "primary_task_loss_weight",
        "stance_auxiliary_loss_weight",
        "posterior_kl_loss_weight",
        "loss_weights",
        "active_views",
        "train_example_count",
        "eval_example_count",
        "monitor_name",
        "monitor_value",
        "head_lr",
        "is_best",
        "best_epoch",
        "best_monitor_value",
        "early_stop_counter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for record in records:
            writer.writerow(_epoch_record_to_csv_row(record, dataset_id=dataset_id, seed=seed, loss_weights=loss_weights))


def _should_run_evaluation() -> bool:
    return True


def _optimizer_group_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    group_lrs: dict[str, float] = {}
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name")
        if isinstance(group_name, str):
            group_lrs[group_name] = float(param_group["lr"])
    return group_lrs


def run_training(
    model: UMCLFiveViewModel,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    primary_task_spec: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: FusionConfig,
    device: torch.device,
    max_epoch: int,
    max_steps: int | None = None,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
    checkpoint_dir: Path | None = None,
    save_best_checkpoint: bool = True,
    log_path: Path | None = None,
    eval_log_path: Path | None = None,
    epoch_log_path: Path | None = None,
    view_mask: "ViewMask | None" = None,  # Add view_mask parameter
    use_amp: bool = False,
    step_log_interval: int = 50,
    on_new_best: Callable[[dict[str, Any]], None] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    enable_early_stop: bool = True,
    train_subset_masker: "DeterministicMixedSubsetMasker | None" = None,
    eval_subset_masker: "DeterministicMixedSubsetMasker | None" = None,
) -> dict[str, Any]:
    epoch_records: list[EpochRecord] = []
    global_step = 0
    last_step_record: TrainStepRecord | None = None
    best_epoch: int | None = None
    best_monitor_value: float | None = None
    early_stop_counter = 0
    stopped_early = False
    stop_reason = "max_epoch_reached"
    selected_monitor_name = validate_monitor_name(config.monitor_name)
    monitor_name = build_eval_monitor_name(primary_task_spec, selected_monitor_name)
    non_blocking_transfer = device.type == "cuda"
    scaler = _build_grad_scaler(device, use_amp)
    amp_enabled = bool(use_amp and device.type == "cuda")
    pending_log_records: list[TrainStepRecord] = []

    if early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1.")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    if eval_log_path is not None:
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        eval_log_path.write_text("", encoding="utf-8")
    if epoch_log_path is not None:
        epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_log_path.write_text("", encoding="utf-8")

    for epoch in range(1, max_epoch + 1):
        collate_fn = getattr(train_loader, "collate_fn", None)
        if hasattr(collate_fn, "set_epoch"):
            collate_fn.set_epoch(epoch)
        model.train()
        LOGGER.info(
            "starting epoch %s/%s (train_batches=%s, eval_batches=%s, amp=%s, device=%s)",
            epoch,
            max_epoch,
            len(train_loader),
            len(eval_loader),
            amp_enabled,
            device,
        )
        sum_losses = _empty_loss_totals_for_view_mask(view_mask)
        total_examples = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            _move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)
            if train_subset_masker is not None:
                train_subset_masker.apply(batch, epoch=epoch)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                output = model(batch, view_mask=view_mask)  # Pass view_mask to model
                losses = compute_total_loss(
                    output,
                    batch,
                    weights=_loss_weights(config),
                    view_mask=view_mask,
                )  # Pass view_mask to loss
            if not _all_losses_finite(losses):
                loss_values = _loss_snapshot(losses)
                if amp_enabled:
                    LOGGER.warning(
                        "non-finite loss detected under AMP at epoch=%s batch=%s step=%s; retrying in full precision. losses=%s",
                        epoch,
                        batch_index,
                        global_step + 1,
                        loss_values,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(device, False):
                        output = model(batch, view_mask=view_mask)
                        losses = compute_total_loss(
                            output,
                            batch,
                            weights=_loss_weights(config),
                            view_mask=view_mask,
                        )
                    if _all_losses_finite(losses):
                        amp_enabled = False
                        scaler = _build_grad_scaler(device, False)
                        LOGGER.warning(
                            "AMP disabled for the remainder of training after full-precision retry succeeded at epoch=%s batch=%s step=%s.",
                            epoch,
                            batch_index,
                            global_step + 1,
                        )
                    else:
                        raise ValueError(
                            f"Non-finite fusion loss after full-precision retry at epoch={epoch} batch={batch_index} "
                            f"step={global_step + 1}: {_loss_snapshot(losses)}"
                        )
                else:
                    raise ValueError(
                        f"Non-finite fusion loss at epoch={epoch} batch={batch_index} step={global_step + 1}: {loss_values}"
                    )
            if scaler.is_enabled():
                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            global_step += 1
            batch_size = len(batch.event_ids)
            total_examples += batch_size
            dataset_ids = sorted(set(batch.dataset_ids))
            if len(dataset_ids) != 1:
                raise ValueError(
                    "Fusion training now assumes single-dataset batches only; "
                    f"got dataset ids {dataset_ids}."
                )
            reconstruction_losses = _reconstruction_loss_snapshot(losses, view_mask)
            record = TrainStepRecord(
                step=global_step,
                epoch=epoch,
                batch_size=batch_size,
                dataset_id=dataset_ids[0],
                active_views=(["content", "stance", "propagation", "evidence", "source"] if view_mask is None else view_mask.get_active_views()),
                total_loss=float(losses["total_loss"].detach().cpu().item()),
                reconstruction_loss=float(losses["reconstruction_loss"].detach().cpu().item()),
                reconstruction_losses=reconstruction_losses,
                task_primary_loss=float(losses["task_primary_loss"].detach().cpu().item()),
                aux_stance_loss=float(losses["aux_stance_loss"].detach().cpu().item()),
                kl_loss=float(losses["kl_loss"].detach().cpu().item()),
            )
            last_step_record = record
            pending_log_records.append(record)
            if log_path is not None and len(pending_log_records) >= max(1, int(step_log_interval)):
                write_train_log(log_path, pending_log_records)
                pending_log_records.clear()
            if global_step % max(1, int(step_log_interval)) == 0:
                LOGGER.info(
                    "epoch=%s step=%s/%s batch=%s/%s total_loss=%.6f reconstruction_loss=%.6f task_primary_loss=%.6f kl_loss=%.6f recon(%s) aux_stance_loss=%.6f active_views=%s",
                    epoch,
                    global_step,
                    max_steps if max_steps is not None else -1,
                    batch_index,
                    len(train_loader),
                    record.total_loss,
                    record.reconstruction_loss,
                    record.task_primary_loss,
                    record.kl_loss,
                    _format_recon_log(record.reconstruction_losses),
                    record.aux_stance_loss,
                    ",".join(record.active_views),
                )
            for key, value in losses.items():
                if key.startswith("_"):
                    continue
                sum_losses[key] = sum_losses.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size
            if max_steps is not None and global_step >= max_steps:
                stop_reason = "max_steps_reached"
                break

        if log_path is not None and pending_log_records:
            write_train_log(log_path, pending_log_records)
            pending_log_records.clear()

        if total_examples == 0:
            raise ValueError("Training loader produced zero examples.")

        train_avg_losses = _average_loss_totals(sum_losses, total_examples)
        LOGGER.info(
            "finished epoch %s train_examples=%s avg_total_loss=%.6f avg_reconstruction_loss=%.6f avg_task_primary_loss=%.6f avg_aux_stance_loss=%.6f avg_kl_loss=%.6f",
            epoch,
            total_examples,
            float(train_avg_losses["total_loss"]),
            float(train_avg_losses["reconstruction_loss"]),
            float(train_avg_losses["task_primary_loss"]),
            float(train_avg_losses["aux_stance_loss"]),
            float(train_avg_losses["kl_loss"]),
        )
        should_eval = _should_run_evaluation()
        eval_example_count: int | None = None
        eval_avg_losses: dict[str, float] | None = None
        monitor_value: float | None = None
        is_best = False

        if should_eval:
            LOGGER.info("running evaluation for epoch %s", epoch)
            eval_summary = run_evaluation(
                model=model,
                loader=eval_loader,
                config=config,
                primary_task_spec=primary_task_spec,
                device=device,
                log_path=None,
                max_batches=None,
                view_mask=view_mask,
                use_amp=amp_enabled,
                subset_masker=eval_subset_masker,
                subset_epoch=0,
            )
            eval_example_count = int(eval_summary["event_count"])
            eval_avg_losses = {key: float(value) for key, value in eval_summary["average_losses"].items()}
            metric_name = str(eval_summary["primary_task"]["metric_name"])
            primary_stats = eval_summary["metrics"][metric_name]
            monitor_value = compute_eval_monitor_value(primary_stats, selected_monitor_name)
            scheduler_active = epoch >= int(config.optimizer["scheduler"]["start_epoch"])
            if scheduler is not None and scheduler_active:
                scheduler.step(monitor_value)
            improvement = best_monitor_value is None or (monitor_value - best_monitor_value) > early_stop_min_delta
            if improvement:
                best_monitor_value = monitor_value
                best_epoch = epoch
                early_stop_counter = 0
                is_best = True
            elif scheduler_active:
                early_stop_counter += 1
            LOGGER.info(
                "completed evaluation for epoch %s monitor=%s value=%.6f best=%.6f best_epoch=%s scheduler_active=%s",
                epoch,
                monitor_name,
                monitor_value,
                float(best_monitor_value) if best_monitor_value is not None else float("nan"),
                best_epoch,
                scheduler_active,
            )
        optimizer_group_lrs = _optimizer_group_lrs(optimizer)
        if "head" not in optimizer_group_lrs:
            raise ValueError("Optimizer group 'head' is required for UMCL training logs.")
        head_lr = float(optimizer_group_lrs["head"])
        epoch_record = EpochRecord(
            epoch=epoch,
            train_example_count=total_examples,
            eval_example_count=eval_example_count,
            active_views=(["content", "stance", "propagation", "evidence", "source"] if view_mask is None else view_mask.get_active_views()),
            train_losses={key: float(value) for key, value in train_avg_losses.items()},
            eval_losses=eval_avg_losses,
            monitor_name=(monitor_name if should_eval else None),
            monitor_value=monitor_value,
            head_lr=head_lr,
            optimizer_group_lrs=optimizer_group_lrs,
            is_best=is_best,
            best_epoch=best_epoch,
            best_monitor_value=best_monitor_value,
            early_stop_counter=early_stop_counter,
        )
        epoch_records.append(epoch_record)
        if epoch_log_path is not None:
            write_epoch_log(epoch_log_path, [epoch_record])
        if should_eval and eval_log_path is not None:
            write_eval_log_csv(
                eval_log_path,
                [epoch_record],
                dataset_id=config.dataset_id or "unknown",
                seed=config.data_loader_seed,
                loss_weights=build_loss_weights(config),
            )

        checkpoint_extra = {
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
            "monitor_name": monitor_name,
            "head_lr": head_lr,
            "optimizer_group_lrs": optimizer_group_lrs,
            "mode": config.mode,
            "active_views": (["content", "stance", "propagation", "evidence", "source"] if view_mask is None else view_mask.get_active_views()),
        }
        checkpoint_path: Path | None = None
        if checkpoint_dir is not None and save_best_checkpoint and should_eval and is_best:
            checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                primary_task_spec=primary_task_spec,
                step=global_step,
                epoch=epoch,
                extra_state=checkpoint_extra,
            )

        if should_eval and is_best and on_new_best is not None:
            on_new_best(
                {
                    "epoch": epoch,
                    "model": model,
                    "eval_summary": eval_summary,
                    "epoch_record": epoch_record,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_saved": checkpoint_path is not None,
                    "monitor_name": monitor_name,
                    "monitor_value": monitor_value,
                }
            )

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch,
                    "model": model,
                    "epoch_record": epoch_record,
                    "eval_summary": eval_summary if should_eval else None,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_saved": checkpoint_path is not None,
                    "monitor_name": monitor_name,
                    "monitor_value": monitor_value,
                }
            )

        if max_epoch is not None and epoch >= max_epoch:
            break
        if enable_early_stop and should_eval and early_stop_counter >= early_stop_patience:
            stopped_early = True
            stop_reason = "early_stop"
            break

    dataset_for_stats = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    cache_stats = dataset_for_stats.cache_stats() if hasattr(dataset_for_stats, "cache_stats") else None
    return {
        "step_records": ([] if last_step_record is None else [last_step_record]),
        "last_step_record": last_step_record,
        "epoch_records": epoch_records,
        "epochs_completed": len(epoch_records),
        "steps_completed": global_step,
        "monitor_name": monitor_name,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "best_checkpoint": (str(checkpoint_dir / "checkpoint_best.pt") if checkpoint_dir is not None and save_best_checkpoint else None),
        "cache_stats": cache_stats,
    }
