from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import json
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from models.fusion.config import FusionConfig
from models.fusion.losses import build_loss_weights, compute_total_loss
from models.fusion.model import UMCLFiveViewModel
from utils.logging_utils import get_logger

if TYPE_CHECKING:
    from models.fusion.subset_sampling import DeterministicMixedSubsetMasker
    from models.fusion.view_mask import ViewMask


LOGGER = get_logger("fusion_eval")


@dataclass
class EvalBatchRecord:
    batch_index: int
    batch_size: int
    dataset_id: str
    active_views: list[str]
    total_loss: float
    reconstruction_loss: float
    task_primary_loss: float
    aux_stance_loss: float
    kl_loss: float
    reconstruction_losses: dict[str, float] = field(default_factory=dict)


def write_eval_log(path: Path, records: Iterable[EvalBatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


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


def _all_losses_finite(losses: dict[str, torch.Tensor]) -> bool:
    for key, value in losses.items():
        if key.startswith("_"):
            continue
        if not torch.isfinite(value).all():
            return False
    return True


def _masked_classification_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_names: list[str],
) -> dict:
    valid = mask > 0
    if not torch.any(valid):
        return {
            "n_valid": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "per_class": {},
        }

    y_true = labels[valid]
    y_pred = logits[valid].argmax(dim=-1)
    n_classes = len(class_names)
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)
    for gold, pred in zip(y_true.view(-1), y_pred.view(-1)):
        confusion[int(gold.item()), int(pred.item())] += 1

    total = int(confusion.sum().item())
    accuracy = float(confusion.diag().sum().item() / total) if total > 0 else None

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in enumerate(class_names):
        tp = float(confusion[class_id, class_id].item())
        fp = float(confusion[:, class_id].sum().item() - tp)
        fn = float(confusion[class_id, :].sum().item() - tp)
        support = int(confusion[class_id, :].sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "n_valid": int(valid.sum().item()),
        "accuracy": accuracy,
        "macro_precision": float(sum(precisions) / n_classes),
        "macro_recall": float(sum(recalls) / n_classes),
        "macro_f1": float(sum(f1s) / n_classes),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _label_distribution_from_stats(stats: dict, class_names: list[str]) -> dict[str, int]:
    per_class = stats.get("per_class") or {}
    return {class_name: int((per_class.get(class_name) or {}).get("support", 0)) for class_name in class_names}


@torch.no_grad()
def run_evaluation(
    model: UMCLFiveViewModel,
    loader: DataLoader,
    config: FusionConfig,
    primary_task_spec: dict[str, object],
    device: torch.device,
    log_path: Path | None = None,
    max_batches: int | None = None,
    view_mask: "ViewMask | None" = None,  # Add view_mask parameter
    use_amp: bool = False,
    subset_masker: "DeterministicMixedSubsetMasker | None" = None,
    subset_epoch: int = 0,
) -> dict:
    model.eval()
    batch_count = 0
    pending_log_records: list[EvalBatchRecord] = []
    active_views = ["content", "stance", "propagation", "evidence", "source"] if view_mask is None else view_mask.get_active_views()
    sum_losses = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "task_primary_loss": 0.0,
        "aux_stance_loss": 0.0,
        "kl_loss": 0.0,
    }
    total_examples = 0
    dataset_id_seen: str | None = None
    logits_primary: list[torch.Tensor] = []
    labels_primary: list[torch.Tensor] = []
    mask_primary: list[torch.Tensor] = []
    non_blocking_transfer = device.type == "cuda"

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        _move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)
        if subset_masker is not None:
            subset_masker.apply(batch, epoch=subset_epoch)
        dataset_ids = sorted(set(batch.dataset_ids))
        if len(dataset_ids) != 1:
            raise ValueError(f"Evaluation now assumes single-dataset batches only; got {dataset_ids}.")
        dataset_id = dataset_ids[0]
        dataset_id_seen = dataset_id_seen or dataset_id

        with _autocast_context(device, use_amp):
            output = model(batch, view_mask=view_mask)  # Pass view_mask
            losses = compute_total_loss(
                output,
                batch,
                weights=build_loss_weights(config),
                view_mask=view_mask,  # Pass view_mask
            )
        if not _all_losses_finite(losses):
            snapshot = {
                key: float(value.detach().cpu().item())
                for key, value in losses.items()
                if not key.startswith("_")
            }
            LOGGER.warning(
                "non-finite evaluation loss detected at batch=%s dataset=%s use_amp=%s losses=%s",
                batch_index,
                dataset_id,
                use_amp,
                snapshot,
            )
            raise ValueError(f"Non-finite evaluation loss detected: {snapshot}")
        batch_size = len(batch.event_ids)
        total_examples += batch_size

        batch_count += 1
        if log_path is not None:
            pending_log_records.append(
                EvalBatchRecord(
                    batch_index=batch_index,
                    batch_size=batch_size,
                    dataset_id=dataset_id,
                    active_views=active_views,
                    total_loss=float(losses["total_loss"].detach().cpu().item()),
                    reconstruction_loss=float(losses["reconstruction_loss"].detach().cpu().item()),
                    reconstruction_losses={
                        key[len("reconstruction_"):-len("_loss")]: float(losses[key].detach().cpu().item())
                        for key in [f"reconstruction_{view}_loss" for view in active_views]
                        if key in losses
                    },
                    task_primary_loss=float(losses["task_primary_loss"].detach().cpu().item()),
                    aux_stance_loss=float(losses["aux_stance_loss"].detach().cpu().item()),
                    kl_loss=float(losses["kl_loss"].detach().cpu().item()),
                )
            )
            if len(pending_log_records) >= 32:
                write_eval_log(log_path, pending_log_records)
                pending_log_records.clear()

        for key, value in losses.items():
            if key.startswith("_"):
                continue
            sum_losses[key] = sum_losses.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size

        logits_primary.append(output.task_outputs["primary_task"]["logits"].detach().cpu())
        labels_primary.append(batch.task_labels["primary_task"].detach().cpu())
        mask_primary.append(batch.task_mask["primary_task"].detach().cpu())

    if log_path is not None:
        if pending_log_records:
            write_eval_log(log_path, pending_log_records)

    if total_examples == 0:
        raise ValueError("Evaluation loader produced zero examples.")

    avg_losses = {name: (value / total_examples if total_examples > 0 else None) for name, value in sum_losses.items()}
    primary_stats = _masked_classification_stats(
        torch.cat(logits_primary, dim=0),
        torch.cat(labels_primary, dim=0),
        torch.cat(mask_primary, dim=0),
        list(primary_task_spec["class_names"]),
    )
    if dataset_id_seen is None:
        raise ValueError("Unable to infer dataset id from evaluation batches.")
    primary_task_summary = {
        "task_name": primary_task_spec["task_name"],
        "metric_name": primary_task_spec["metric_name"],
        "class_names": list(primary_task_spec["class_names"]),
        "n_valid": int(primary_stats["n_valid"]),
        "label_distribution": _label_distribution_from_stats(primary_stats, list(primary_task_spec["class_names"])),
    }

    return {
        "dataset_id": dataset_id_seen,
        "event_count": total_examples,
        "average_losses": avg_losses,
        "primary_task": primary_task_summary,
        "metrics": {str(primary_task_spec["metric_name"]): primary_stats},
        "batch_count": batch_count,
        "max_batches": max_batches,
    }
