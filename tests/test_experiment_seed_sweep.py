"""Checkpoint manifest and five-seed experiment regression tests."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import experiment_seed_sweep as sweep


class UMCLSeedSweepTests(unittest.TestCase):
    @staticmethod
    def _write_candidate_checkpoint(path: Path, dataset_id: str, seed: int) -> None:
        torch.save(
            {
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "primary_task_spec": {
                    "task_name": "veracity",
                    "metric_name": "veracity_macro_f1",
                    "class_names": ["false", "true"],
                },
                "fusion_config": {
                    "dataset_id": dataset_id,
                    "mode": "full",
                    "fusion_variant": "poe",
                    "active_views": "content,stance,propagation",
                    "data_loader_seed": seed,
                    "max_epoch": 10,
                    "candidate_epochs": 4,
                },
                "active_views": list(sweep.STANDARD_ACTIVE_VIEWS),
                "epoch": 7,
                "best_epoch": 7,
                "selection_scope": sweep.BEST_SELECTION_SCOPE_CANDIDATE_EPOCHS,
                "selection_rule": sweep.BEST_SELECTION_RULE_RAW_MAX,
                "candidate_epoch_start": 7,
                "candidate_epoch_end": 10,
                "best_selection_min_delta": 0.0,
            },
            path,
        )

    def _args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_id="PHEME",
            pretrained_model=None,
            device="cpu",
            data_root=str(root / "data"),
            run_root=str(root / "runs"),
            manifest=str(root / "results" / "standard_umcl_checkpoints.csv"),
        )

    def _create_completed_run(
        self,
        command: list[str],
        *,
        suffix: str = "completed",
        auto_test_ran: bool = True,
        active_views: list[str] | None = None,
        content_model: str = "xlm-roberta-base",
        stance_model: str = "xlm-roberta-base",
        create_checkpoint: bool = True,
    ) -> Path:
        dataset_id = command[command.index("--dataset_id") + 1]
        seed = int(command[command.index("--seed") + 1])
        run_root = Path(command[command.index("--run_root") + 1])
        run_dir = run_root / dataset_id / f"seed_{seed}" / f"run_{seed}_{suffix}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint_best.pt"
        if create_checkpoint:
            self._write_candidate_checkpoint(checkpoint, dataset_id, seed)
        summary = {
            "run": {"dataset_id": dataset_id, "seed": seed, "run_dir": str(run_dir)},
            "data": {
                "active_views": active_views
                if active_views is not None
                else ["content", "stance", "propagation"]
            },
            "hyperparameters": {
                "pretrained_models": {
                    "content": content_model,
                    "stance": stance_model,
                }
            },
            "training": {"best_checkpoint": str(checkpoint), "best_epoch": 7},
            "testing": {"auto_test_ran": auto_test_ran, "best_epoch": 7},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run_dir

    def test_runs_five_seeds_with_legacy_entrypoint_and_appends_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            args.data_root = "data"
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                commands.append(list(command))
                self.assertTrue(kwargs["check"])
                self.assertEqual(Path(kwargs["cwd"]), sweep.REPOSITORY_ROOT)
                self._create_completed_run(list(command))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
                sweep.run_seed_sweep(args)

            self.assertEqual(
                [int(command[command.index("--seed") + 1]) for command in commands],
                list(sweep.SEEDS),
            )
            for command in commands:
                self.assertEqual(Path(command[1]), sweep.TRAIN_ENTRYPOINT)
                checkpoint_index = command.index("--checkpoint")
                self.assertEqual(command[checkpoint_index + 1], "true")
                data_root_index = command.index("--data_root")
                self.assertEqual(command[data_root_index + 1], "data")

            manifest = Path(args.manifest)
            lines = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines.count(",".join(sweep.MANIFEST_FIELDS)), 1)
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["seed"]) for row in rows], list(sweep.SEEDS))
            self.assertTrue(all(row["pretrained_model"] == "xlm-roberta-base" for row in rows))
            self.assertTrue(all(row["checkpoint_path"].startswith("../runs/") for row in rows))

    def test_skips_existing_checkpoint_and_replaces_stale_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            manifest = Path(args.manifest)
            manifest.parent.mkdir(parents=True)
            valid_checkpoint = root / "runs" / "PHEME" / "seed_2" / "old" / "checkpoint_best.pt"
            valid_checkpoint.parent.mkdir(parents=True)
            self._write_candidate_checkpoint(valid_checkpoint, "PHEME", 2)
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sweep.MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_id": "PHEME",
                        "seed": 2,
                        "checkpoint_path": "../runs/PHEME/seed_2/old/checkpoint_best.pt",
                        "pretrained_model": "xlm-roberta-base",
                    }
                )
                writer.writerow(
                    {
                        "dataset_id": "PHEME",
                        "seed": 4,
                        "checkpoint_path": "../runs/PHEME/seed_4/missing/checkpoint_best.pt",
                        "pretrained_model": "xlm-roberta-base",
                    }
                )
            invoked_seeds: list[int] = []

            def fake_run(command, **_kwargs):
                seed = int(command[command.index("--seed") + 1])
                invoked_seeds.append(seed)
                self._create_completed_run(list(command), suffix="replacement")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
                sweep.run_seed_sweep(args)

            self.assertEqual(invoked_seeds, [4, 8, 16, 32])
            rows = sweep._load_manifest(manifest)
            self.assertEqual(len(rows), 5)
            self.assertEqual(len({(row["dataset_id"], row["seed"]) for row in rows}), 5)
            seed_4 = next(row for row in rows if int(row["seed"]) == 4)
            self.assertIn("run_4_replacement", seed_4["checkpoint_path"])

    def test_existing_full_run_validation_best_checkpoint_is_retrained(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            manifest = Path(args.manifest)
            manifest.parent.mkdir(parents=True)
            old_checkpoint = (
                root / "runs" / "PHEME" / "seed_2" / "old" / "checkpoint_best.pt"
            )
            old_checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "fusion_config": {
                        "dataset_id": "PHEME",
                        "data_loader_seed": 2,
                        "active_views": "content,stance,propagation",
                        "max_epoch": 10,
                        "candidate_epochs": 4,
                    },
                    "active_views": list(sweep.STANDARD_ACTIVE_VIEWS),
                    "epoch": 2,
                    "best_epoch": 2,
                },
                old_checkpoint,
            )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sweep.MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_id": "PHEME",
                        "seed": 2,
                        "checkpoint_path": "../runs/PHEME/seed_2/old/checkpoint_best.pt",
                        "pretrained_model": "xlm-roberta-base",
                    }
                )
            invoked: list[int] = []

            def fake_run(command, **_kwargs):
                seed = int(command[command.index("--seed") + 1])
                invoked.append(seed)
                self._create_completed_run(list(command), suffix="candidate")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
                sweep.run_seed_sweep(args)

            self.assertEqual(invoked, list(sweep.SEEDS))
            rows = sweep._load_manifest(manifest)
            seed_2 = next(row for row in rows if int(row["seed"]) == 2)
            self.assertIn("run_2_candidate", seed_2["checkpoint_path"])

    def test_manifest_rejects_wrong_header_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            manifest.write_text("dataset_id,seed,checkpoint_path\nPHEME,2,a.pt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header must be exactly"):
                sweep._load_manifest(manifest)

            manifest.write_text(
                ",".join(sweep.MANIFEST_FIELDS)
                + "\nPHEME,2,a.pt,xlm-roberta-base\nPHEME,2,b.pt,xlm-roberta-base\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate dataset/seed"):
                sweep._load_manifest(manifest)

    def test_data_root_argument_reuses_existing_fingerprint_path_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            metadata_path = data_root / "content_view" / "PHEME" / "full" / "artifact_metadata.json"
            metadata_path.parent.mkdir(parents=True)

            def write_metadata(input_root: str) -> None:
                metadata_path.write_text(
                    json.dumps(
                        {
                            "artifacts": {
                                "event_index": {
                                    "fingerprint_input": {"input": {"input_root": input_root}}
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch.object(sweep, "REPOSITORY_ROOT", root):
                write_metadata("data/preprocessed")
                relative = sweep._artifact_compatible_data_root_argument(
                    requested_value=str(data_root),
                    resolved_data_root=data_root,
                    dataset_id="PHEME",
                )
                self.assertEqual(relative, "data")

                write_metadata(str(data_root / "preprocessed"))
                absolute = sweep._artifact_compatible_data_root_argument(
                    requested_value="data",
                    resolved_data_root=data_root,
                    dataset_id="PHEME",
                )
                self.assertEqual(Path(absolute), data_root)

    def test_summary_validation_rejects_invalid_test_views_checkpoint_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = sweep._training_command(
                dataset_id="PHEME",
                seed=2,
                pretrained_model=None,
                device="cpu",
                data_root=root / "data",
                run_root=root / "runs",
            )
            run_dir = self._create_completed_run(command)
            base_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = root / "results" / "standard_umcl_checkpoints.csv"

            cases = [
                ("test did not run", {"testing": {"auto_test_ran": False}}, "test did not complete"),
                (
                    "wrong views",
                    {"data": {"active_views": ["content", "stance"]}},
                    "exactly C/S/P",
                ),
                (
                    "different models",
                    {
                        "hyperparameters": {
                            "pretrained_models": {"content": "model-a", "stance": "model-b"}
                        }
                    },
                    "different pretrained models",
                ),
                ("missing checkpoint", {"training": {"best_checkpoint": None}}, "did not report"),
            ]
            for label, replacement, pattern in cases:
                with self.subTest(label=label):
                    summary = {**base_summary, **replacement}
                    with self.assertRaisesRegex((ValueError, FileNotFoundError), pattern):
                        sweep._manifest_row_from_summary(
                            summary=summary,
                            run_dir=run_dir,
                            dataset_id="PHEME",
                            seed=2,
                            manifest_path=manifest,
                        )

    def test_missing_summary_pretrained_models_fall_back_to_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = sweep._training_command(
                dataset_id="PHEME",
                seed=2,
                pretrained_model=None,
                device="cpu",
                data_root=root / "data",
                run_root=root / "runs",
            )
            run_dir = self._create_completed_run(command)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["hyperparameters"].pop("pretrained_models")
            configured = SimpleNamespace(model_name=lambda _mode: "configured-model")

            with mock.patch.object(sweep, "resolve_encoder_config", return_value=configured) as resolver:
                row = sweep._manifest_row_from_summary(
                    summary=summary,
                    run_dir=run_dir,
                    dataset_id="PHEME",
                    seed=2,
                    manifest_path=root / "results" / "standard_umcl_checkpoints.csv",
                )

            self.assertEqual(row["pretrained_model"], "configured-model")
            self.assertEqual(
                [call.args[0] for call in resolver.call_args_list],
                ["content", "stance"],
            )

    def test_missing_summary_models_use_explicit_pretrained_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = sweep._training_command(
                dataset_id="PHEME",
                seed=2,
                pretrained_model="custom-model",
                device="cpu",
                data_root=root / "data",
                run_root=root / "runs",
            )
            run_dir = self._create_completed_run(command)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["hyperparameters"] = {}

            with mock.patch.object(sweep, "resolve_encoder_config") as resolver:
                row = sweep._manifest_row_from_summary(
                    summary=summary,
                    run_dir=run_dir,
                    dataset_id="PHEME",
                    seed=2,
                    manifest_path=root / "results" / "standard_umcl_checkpoints.csv",
                    pretrained_model_override="custom-model",
                )

            resolver.assert_not_called()
            self.assertEqual(row["pretrained_model"], "custom-model")

    def test_subprocess_failure_or_ambiguous_new_run_does_not_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            manifest = Path(args.manifest)
            with mock.patch.object(
                sweep.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["python", "train_umcl.py"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    sweep.run_seed_sweep(args)
            self.assertFalse(manifest.exists())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            manifest = Path(args.manifest)

            def fake_ambiguous_run(command, **_kwargs):
                self._create_completed_run(list(command), suffix="one")
                self._create_completed_run(list(command), suffix="two")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sweep.subprocess, "run", side_effect=fake_ambiguous_run):
                with self.assertRaisesRegex(RuntimeError, "exactly one new run"):
                    sweep.run_seed_sweep(args)
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
