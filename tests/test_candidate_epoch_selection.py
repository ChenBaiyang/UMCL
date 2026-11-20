from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models.fusion.config import FusionConfig
from models.fusion.training import (
    BEST_SELECTION_RULE_RAW_MAX,
    BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
    run_training,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, _batch, view_mask=None):
        return self.weight


class _OneBatchLoader:
    def __init__(self) -> None:
        self.dataset = SimpleNamespace()
        self.collate_fn = None
        self.batch = SimpleNamespace(
            modalities={},
            modality_mask={},
            task_labels={},
            task_mask={},
            event_ids=["event-1"],
            dataset_ids=["PHEME"],
        )

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


class _RecordingScheduler:
    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float) -> None:
        self.values.append(float(value))


class CandidateEpochSelectionTests(unittest.TestCase):
    def test_validation_and_scheduler_continue_before_candidate_selection(self) -> None:
        config = FusionConfig(dataset_id="PHEME", mode="full")
        config.data_loader_seed = 2
        config.max_epoch = 5
        config.candidate_epochs = 3
        config.optimizer["scheduler"]["start_epoch"] = 1
        model = _TinyModel()
        optimizer = torch.optim.SGD(
            [{"params": list(model.parameters()), "lr": 0.01, "name": "head"}]
        )
        scheduler = _RecordingScheduler()
        monitor_values = iter([0.99, 0.98, 0.5, 0.50005, 0.50005])
        new_best_epochs: list[int] = []

        def fake_loss(_output, _batch, **_kwargs):
            total = model.weight.square().mean()
            zero = model.weight.sum() * 0.0
            return {
                "total_loss": total,
                "reconstruction_loss": zero,
                "task_primary_loss": zero,
                "aux_stance_loss": zero,
                "kl_loss": zero,
            }

        def fake_evaluation(**_kwargs):
            value = next(monitor_values)
            return {
                "event_count": 1,
                "average_losses": {"total_loss": 0.0},
                "primary_task": {"metric_name": "veracity_macro_f1"},
                "metrics": {
                    "veracity_macro_f1": {"macro_f1": value, "accuracy": value}
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("models.fusion.training.compute_total_loss", side_effect=fake_loss),
                mock.patch("models.fusion.training.run_evaluation", side_effect=fake_evaluation),
            ):
                result = run_training(
                    model=model,
                    train_loader=_OneBatchLoader(),
                    eval_loader=_OneBatchLoader(),
                    primary_task_spec={
                        "metric_name": "veracity_macro_f1",
                        "class_names": ["false", "true"],
                    },
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    device=torch.device("cpu"),
                    max_epoch=5,
                    early_stop_min_delta=1e-4,
                    checkpoint_dir=Path(tmp),
                    on_new_best=lambda event: new_best_epochs.append(int(event["epoch"])),
                    enable_early_stop=False,
                    best_selection_start_epoch=3,
                    best_selection_min_delta=0.0,
                    best_selection_scope=BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                    best_selection_rule=BEST_SELECTION_RULE_RAW_MAX,
                )

            records = result["epoch_records"]
            self.assertEqual(len(records), 5)
            self.assertEqual(len(scheduler.values), 5)
            self.assertEqual([record.is_best for record in records[:2]], [False, False])
            self.assertEqual([record.best_epoch for record in records[:2]], [None, None])
            self.assertEqual(new_best_epochs, [3, 4])
            self.assertEqual(result["best_epoch"], 4)
            self.assertAlmostEqual(result["best_monitor_value"], 0.50005)
            checkpoint = torch.load(Path(tmp) / "checkpoint_best.pt", map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 4)
            self.assertEqual(checkpoint["best_epoch"], 4)
            self.assertEqual(
                checkpoint["selection_scope"], BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS
            )
            self.assertEqual(checkpoint["selection_rule"], BEST_SELECTION_RULE_RAW_MAX)
            self.assertEqual(checkpoint["candidate_epoch_start"], 3)
            self.assertEqual(checkpoint["candidate_epoch_end"], 5)


if __name__ == "__main__":
    unittest.main()
