from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import experiment_mixed_subsets as experiment


TASK_SPEC = {
    "task_name": "veracity",
    "metric_name": "veracity_macro_f1",
    "class_names": ["false", "true"],
}


def _checkpoint_payload(*, dataset_id: str = "PHEME", seed: int = 2, active_views=None) -> dict:
    return {
        "model_state_dict": {"weight": torch.tensor([3.0])},
        "fusion_config": {
            "dataset_id": dataset_id,
            "mode": "full",
            "fusion_variant": "poe",
            "active_views": "content,stance,propagation",
            "data_loader_seed": seed,
            "batch_size": 2,
            "max_epoch": 10,
            "candidate_epochs": 4,
            "optimizer": {},
        },
        "primary_task_spec": TASK_SPEC,
        "active_views": active_views or ["content", "stance", "propagation"],
        "epoch": 7,
        "step": 11,
        "best_epoch": 7,
        "selection_scope": experiment.BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
        "selection_rule": experiment.BEST_SELECTION_RULE_RAW_MAX,
        "candidate_epoch_start": 7,
        "candidate_epoch_end": 10,
        "best_selection_min_delta": 0.0,
    }


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class CompleteOnlyCheckpointTests(unittest.TestCase):
    def test_manifest_resolves_relative_checkpoint_and_validates_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "seed_2.pt"
            checkpoint.parent.mkdir()
            torch.save(_checkpoint_payload(), checkpoint)
            manifest = root / "standard.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_id", "seed", "checkpoint_path"])
                writer.writeheader()
                writer.writerow(
                    {"dataset_id": "PHEME", "seed": 2, "checkpoint_path": "checkpoints/seed_2.pt"}
                )
            records = experiment._load_checkpoint_manifest(
                manifest,
                datasets=["PHEME"],
                seeds=[2],
                require_complete_grid=False,
            )
            record = records[("PHEME", 2)]
            self.assertEqual(Path(record["checkpoint_path"]), checkpoint.resolve())
            self.assertEqual(len(record["checkpoint_sha256"]), 64)
            self.assertEqual(record["active_views"], list(experiment.CORE_VIEWS))

    def test_checkpoint_rejects_extra_view(self) -> None:
        payload = _checkpoint_payload(
            active_views=["content", "stance", "propagation", "evidence"]
        )
        with self.assertRaisesRegex(ValueError, "exactly C/S/P"):
            experiment._validate_checkpoint_payload(
                Path("bad.pt"), payload, dataset_id="PHEME", seed=2
            )

    def test_checkpoint_rejects_wrong_dataset_and_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            experiment._validate_checkpoint_payload(
                Path("bad_dataset.pt"),
                _checkpoint_payload(dataset_id="Weibo", seed=2),
                dataset_id="PHEME",
                seed=2,
            )

    def test_checkpoint_rejects_old_or_out_of_range_selection(self) -> None:
        old_payload = _checkpoint_payload()
        old_payload.pop("selection_scope")
        with self.assertRaisesRegex(ValueError, "not candidate-epoch selected"):
            experiment._validate_checkpoint_payload(
                Path("old.pt"), old_payload, dataset_id="PHEME", seed=2
            )

        out_of_range = _checkpoint_payload()
        out_of_range["epoch"] = 6
        out_of_range["best_epoch"] = 6
        with self.assertRaisesRegex(ValueError, "outside candidate range"):
            experiment._validate_checkpoint_payload(
                Path("outside.pt"), out_of_range, dataset_id="PHEME", seed=2
            )
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            experiment._validate_checkpoint_payload(
                Path("bad_seed.pt"),
                _checkpoint_payload(dataset_id="PHEME", seed=4),
                dataset_id="PHEME",
                seed=2,
            )

    def test_manifest_rejects_missing_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "standard.csv"
            manifest.write_text(
                "dataset_id,seed,checkpoint_path\nPHEME,2,missing.pt\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint is missing"):
                experiment._load_checkpoint_manifest(
                    manifest,
                    datasets=["PHEME"],
                    seeds=[2],
                    require_complete_grid=False,
                )

    def test_manifest_complete_grid_requires_all_ten_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "standard.csv"
            rows = ["dataset_id,seed,checkpoint_path"]
            for dataset_id in experiment.DEFAULT_DATASETS:
                for seed in experiment.DEFAULT_SEEDS:
                    if (dataset_id, seed) != ("Weibo", 32):
                        rows.append(f"{dataset_id},{seed},unused.pt")
            manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly cover the standard ten runs"):
                experiment._load_checkpoint_manifest(
                    manifest,
                    datasets=list(experiment.DEFAULT_DATASETS),
                    seeds=list(experiment.DEFAULT_SEEDS),
                    require_complete_grid=True,
                )

    def test_manifest_rejects_duplicate_dataset_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "seed_2.pt"
            torch.save(_checkpoint_payload(), checkpoint)
            manifest = root / "standard.csv"
            manifest.write_text(
                "dataset_id,seed,checkpoint_path\n"
                "PHEME,2,seed_2.pt\n"
                "PHEME,2,seed_2.pt\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                experiment._load_checkpoint_manifest(
                    manifest,
                    datasets=["PHEME"],
                    seeds=[2],
                    require_complete_grid=False,
                )

    def test_manifest_rejects_checkpoint_reuse_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "seed_2.pt"
            torch.save(_checkpoint_payload(), checkpoint)
            manifest = root / "standard.csv"
            manifest.write_text(
                "dataset_id,seed,checkpoint_path\n"
                "PHEME,2,seed_2.pt\n"
                "PHEME,4,seed_2.pt\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be reused"):
                experiment._load_checkpoint_manifest(
                    manifest,
                    datasets=["PHEME"],
                    seeds=[2, 4],
                    require_complete_grid=False,
                )

    def test_task_compatibility_rejects_class_space_mismatch(self) -> None:
        mismatched = {**TASK_SPEC, "class_names": ["false", "true", "unverified"]}
        with self.assertRaisesRegex(ValueError, "class names"):
            experiment._validate_task_compatibility(TASK_SPEC, mismatched)

    def test_global_preflight_runtime_checks_every_requested_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = {
                ("PHEME", 2): {"dataset_id": "PHEME", "seed": 2},
                ("PHEME", 4): {"dataset_id": "PHEME", "seed": 4},
            }
            specialist_rows = [
                {
                    "dataset_id": "PHEME",
                    "seed": seed,
                    "subset": subset,
                }
                for seed in (2, 4)
                for subset in experiment.SUBSET_ORDER
            ]
            args = argparse.Namespace(
                specialist_results=str(Path(tmp) / "specialist.csv"),
                complete_only_checkpoint_manifest=str(Path(tmp) / "manifest.csv"),
                require_complete_grid=False,
                run_root=str(Path(tmp) / "runs"),
            )

            def fake_runtime(_args, record):
                return {"dataset_id": record["dataset_id"], "seed": record["seed"]}

            with (
                mock.patch.object(experiment, "_specialist_seed_rows", return_value=specialist_rows),
                mock.patch.object(experiment, "_validate_specialist_inputs"),
                mock.patch.object(experiment, "_load_checkpoint_manifest", return_value=records),
                mock.patch.object(
                    experiment,
                    "_preflight_checkpoint_runtime",
                    side_effect=fake_runtime,
                ) as runtime_check,
            ):
                _, checked_records = experiment._preflight_inputs(
                    args,
                    datasets=["PHEME"],
                    regimes=["complete_only"],
                    seeds=[2, 4],
                )

            self.assertEqual(runtime_check.call_count, 2)
            self.assertEqual(
                [checked_records[key]["runtime_validation"]["seed"] for key in sorted(checked_records)],
                [2, 4],
            )

    def test_complete_only_reuses_one_model_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "standard.pt"
            torch.save(_checkpoint_payload(), checkpoint)
            record = {
                "dataset_id": "PHEME",
                "seed": 2,
                "active_views": list(experiment.CORE_VIEWS),
                "epoch": 7,
                "step": 11,
                "selection_scope": experiment.BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                "selection_rule": experiment.BEST_SELECTION_RULE_RAW_MAX,
                "candidate_epoch_start": 7,
                "candidate_epoch_end": 10,
                "best_selection_min_delta": 0.0,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": experiment._sha256_file(checkpoint),
                "model_source": "standard_umcl_checkpoint_manifest",
                "pretrained_model": None,
                "pretrained_model_source": "configured_default",
                "manifest_path": str(root / "standard.csv"),
            }
            args = argparse.Namespace(
                run_root=str(root / "output"),
                force=True,
                device="cpu",
                cuda_index=0,
                amp="false",
                data_root=str(root / "data"),
                batch_size=None,
                num_workers=0,
                prefetch_factor=None,
                disable_persistent_workers=True,
                disable_pin_memory=True,
            )
            model = TinyModel()
            model_ids: list[int] = []
            masks: list[set[str]] = []

            def fake_masked_test(**kwargs):
                model_ids.append(id(kwargs["model"]))
                masks.append(set(kwargs["view_mask"].active_views))
                return {
                    "active_views": kwargs["view_mask"].get_active_views(),
                    "event_count": 3,
                    "valid_prediction_count": 3,
                    "macro_f1": 0.8,
                    "accuracy": 0.75,
                    "posterior_uncertainty": None,
                }

            specialist = {
                "macro_f1": 0.8,
                "accuracy": 0.75,
                "run_dir": "results/result_view_contribution.csv",
            }
            with (
                mock.patch.object(experiment, "ensure_model_inputs", return_value=[]),
                mock.patch.object(
                    experiment,
                    "build_eval_loader",
                    return_value=(object(), 3, ["test"], {"content_surface": 1}, {}, TASK_SPEC),
                ),
                mock.patch.object(experiment, "create_fusion_model", return_value=model),
                mock.patch.object(experiment, "_run_masked_test", side_effect=fake_masked_test),
                mock.patch.object(experiment, "run_training") as run_training_mock,
                mock.patch.object(experiment, "build_train_loader") as train_loader_mock,
                mock.patch.object(experiment, "build_hierarchical_optimizer") as optimizer_mock,
            ):
                summary = experiment._run_complete_only(
                    args, "PHEME", 2, record, specialist
                )

            run_training_mock.assert_not_called()
            train_loader_mock.assert_not_called()
            optimizer_mock.assert_not_called()
            self.assertEqual(len(model_ids), 7)
            self.assertEqual(len(set(model_ids)), 1)
            self.assertEqual(
                masks,
                [set(views) for _, views in experiment.SUBSET_SPECS],
            )
            self.assertEqual(float(model.weight.detach().item()), 3.0)
            self.assertTrue(
                all("macro_f1" in result and "accuracy" in result
                    for result in summary["fixed_subset_tests"].values())
            )
            self.assertFalse(summary["training"]["performed"])
            self.assertEqual(summary["protocol"]["parameter_updates"], 0)
            self.assertEqual(summary["source_checkpoint"]["checkpoint_sha256"], record["checkpoint_sha256"])

    def test_mixed_reuses_candidate_best_state_for_all_seven_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_args = SimpleNamespace(
                batch_size=2,
                max_epoch=5,
                candidate_epochs=3,
                max_steps=None,
                early_stop_patience=5,
                early_stop_min_delta=1e-4,
                num_workers=0,
                prefetch_factor=None,
                disable_persistent_workers=True,
                disable_pin_memory=True,
                log_every_steps=50,
                splits=None,
                eval_splits=None,
            )
            config = SimpleNamespace(optimizer={"scheduler": {"start_epoch": 1}})
            optimizer_config = {
                "backbone": {"lr": 1e-5, "weight_decay": 0.0},
                "top_layer": {"lr": 1e-5, "weight_decay": 0.0},
                "head": {"lr": 1e-4, "weight_decay": 0.0},
                "scheduler": {
                    "factor": 0.5,
                    "patience": 2,
                    "min_lr": 1e-6,
                    "threshold": 1e-4,
                    "threshold_mode": "abs",
                },
            }
            args = argparse.Namespace(
                run_root=str(root / "output"),
                force=True,
                data_root=str(root / "data"),
                device="cpu",
                cuda_index=0,
                amp="false",
                checkpoint="false",
                splits=None,
                eval_splits=None,
            )
            model = TinyModel()
            model_ids: list[int] = []
            observed_weights: list[float] = []

            def fake_training(**kwargs):
                self.assertEqual(kwargs["best_selection_start_epoch"], 3)
                self.assertEqual(kwargs["best_selection_min_delta"], 0.0)
                self.assertEqual(
                    kwargs["best_selection_scope"],
                    experiment.BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                )
                self.assertEqual(
                    kwargs["best_selection_rule"], experiment.BEST_SELECTION_RULE_RAW_MAX
                )
                with torch.no_grad():
                    kwargs["model"].weight.fill_(4.0)
                kwargs["on_new_best"](
                    {
                        "epoch": 4,
                        "step": 40,
                        "model": kwargs["model"],
                        "epoch_record": SimpleNamespace(epoch=4, monitor_value=0.8),
                    }
                )
                with torch.no_grad():
                    kwargs["model"].weight.fill_(5.0)
                return {
                    "best_epoch": 4,
                    "best_monitor_value": 0.8,
                    "steps_completed": 50,
                    "selection_scope": experiment.BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                    "selection_rule": experiment.BEST_SELECTION_RULE_RAW_MAX,
                }

            def fake_masked_test(**kwargs):
                model_ids.append(id(kwargs["model"]))
                observed_weights.append(float(kwargs["model"].weight.detach().item()))
                return {
                    "event_count": 3,
                    "valid_prediction_count": 3,
                    "macro_f1": 0.8,
                    "accuracy": 0.75,
                }

            loader_result = (
                object(),
                3,
                ["train"],
                {"content_surface": 1},
                {},
                TASK_SPEC,
                {},
            )
            eval_result = (
                object(),
                3,
                ["validation"],
                {"content_surface": 1},
                {},
                TASK_SPEC,
            )
            test_result = (
                object(),
                3,
                ["test"],
                {"content_surface": 1},
                {},
                TASK_SPEC,
            )
            with (
                mock.patch.object(
                    experiment,
                    "_prepare_mixed_run",
                    return_value=(run_args, config, optimizer_config, {"version": 3}),
                ),
                mock.patch.object(experiment, "ensure_model_inputs", return_value=[]),
                mock.patch.object(experiment, "build_train_loader", return_value=loader_result),
                mock.patch.object(experiment, "build_eval_loader", return_value=eval_result),
                mock.patch.object(
                    experiment, "build_test_loader_with_fallback", return_value=test_result
                ),
                mock.patch.object(experiment, "create_fusion_model", return_value=model),
                mock.patch.object(experiment, "build_hierarchical_optimizer", return_value=object()),
                mock.patch.object(
                    experiment.torch.optim.lr_scheduler,
                    "ReduceLROnPlateau",
                    return_value=object(),
                ),
                mock.patch.object(experiment, "run_training", side_effect=fake_training),
                mock.patch.object(experiment, "_run_masked_test", side_effect=fake_masked_test),
            ):
                summary = experiment._run_mixed(args, "PHEME", 2)

            self.assertEqual(summary["selection"]["epoch"], 4)
            self.assertEqual(len(model_ids), 7)
            self.assertEqual(len(set(model_ids)), 1)
            self.assertEqual(observed_weights, [4.0] * 7)
            self.assertEqual(float(model.weight.detach().item()), 4.0)


if __name__ == "__main__":
    unittest.main()
