"""Shared runtime operations for training, evaluation, and experiments."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch

from models.fusion.config import normalize_optimizer_config


def resolve_device(choice: str, cuda_index: int = 0) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" or torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def resolve_amp(choice: str, device: torch.device) -> bool:
    return choice != "false" and device.type == "cuda"


def resolve_checkpoint_choice(choice: str) -> bool:
    return choice == "true"


def parse_split_arg(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()] or None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_dirstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_optimizer_config(args, config) -> dict[str, Any]:
    optimizer = normalize_optimizer_config(config.optimizer)
    if args is not None:
        for group, fields in {
            "backbone": ("lr", "weight_decay"),
            "top_layer": ("lr", "weight_decay"),
            "head": ("lr", "weight_decay"),
            "scheduler": ("patience", "factor", "min_lr", "start_epoch"),
        }.items():
            for field in fields:
                value = getattr(args, f"optimizer_{group}_{field}")
                if value is not None:
                    optimizer[group][field] = value
    return optimizer


def scheduler_params(optimizer: dict[str, Any]) -> dict[str, Any]:
    scheduler = optimizer["scheduler"]
    return {"mode": "max", "factor": float(scheduler["factor"]),
            "patience": int(scheduler["patience"]), "min_lr": float(scheduler["min_lr"])}


def move_batch_to_device(batch, device: torch.device, *, non_blocking: bool = False) -> None:
    for payload in [*batch.modalities.values(), batch.modality_mask, batch.task_labels, batch.task_mask]:
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.to(device, non_blocking=non_blocking)


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def all_losses_finite(losses: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all() for key, value in losses.items() if not key.startswith("_"))
