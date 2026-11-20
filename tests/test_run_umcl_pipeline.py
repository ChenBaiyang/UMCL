from __future__ import annotations

import argparse
import inspect
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_umcl_pipeline as pipeline
from config.resolver import resolve_encoder_config
from utils.jsonl import write_json


def _optimizer_config() -> dict:
    return {
        "backbone": {"lr": 2e-5, "weight_decay": 0.01},
        "top_layer": {"lr": 5e-5, "weight_decay": 0.01},
        "head": {"lr": 1e-4, "weight_decay": 1e-4},
        "scheduler": {"factor": 0.5, "patience": 2, "min_lr": 1e-6, "start_epoch": 5},
    }


class _FakeConfig:
    def __init__(self) -> None:
        self.dataset_id = "PHEME"
        self.mode = "full"
        self.optimizer = _optimizer_config()
        self.active_views = "content,stance"
        self.views = ["content", "stance", "propagation", "evidence", "source"]
        self.sample_latent_during_training = None
        self.batch_size = 4
        self.max_epoch = 3
        self.early_stop_patience = 2
        self.data_loader_seed = 2
        self.resolved_overrides = {}
        self.monitor_name = "macro_f1"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class _FakeModel:
    def to(self, _device):
        return self


@dataclass
class _EpochRecord:
    epoch: int = 2
    monitor_value: float = 0.75


class StandardPipelineTests(unittest.TestCase):
    def test_standard_test_call_uses_only_required_evaluation_parameters(self) -> None:
        source = inspect.getsource(pipeline._run_pipeline)
        self.assertNotIn("efficiency=", source)
        self.assertNotIn("warmup_runs=", source)
        self.assertNotIn("benchmark_runs=", source)

    def test_parser_exposes_only_core_options(self) -> None:
        option_dests = {
            action.dest
            for action in pipeline.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            option_dests,
            {"dataset_id", "pretrained_model", "seed", "device", "data_root", "run_root"},
        )
        args = pipeline.parse_args(["--dataset_id", "PHEME"])
        self.assertEqual(args.seed, 2)
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.data_root, "data")
        self.assertEqual(args.run_root, "runs")
        self.assertIsNone(args.pretrained_model)

    def test_pipeline_uses_best_validation_checkpoint_and_tests_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(
                dataset_id="PHEME",
                pretrained_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                seed=7,
                device="cpu",
                data_root=str(root / "data"),
                run_root=str(root / "runs"),
            )
            train_loader = object()
            validation_loader = object()
            feature_dims = {"content_surface": 9, "stance_reply_embedding": 384}
            primary_task_spec = {"task_name": "veracity", "class_names": ["true", "false"]}
            captured_model_name: list[str] = []

            def fake_training(**kwargs):
                self.assertEqual(kwargs["max_epoch"], 3)
                self.assertEqual(kwargs["early_stop_patience"], 2)
                self.assertTrue(kwargs["enable_early_stop"])
                self.assertTrue(kwargs["save_best_checkpoint"])
                self.assertNotIn("best_selection_start_epoch", kwargs)
                self.assertNotIn("best_selection_scope", kwargs)
                checkpoint_path = kwargs["checkpoint_dir"] / "checkpoint_best.pt"
                checkpoint_path.write_bytes(b"checkpoint")
                kwargs["eval_log_path"].write_text("validation\n", encoding="utf-8")
                kwargs["epoch_log_path"].write_text("{}\n", encoding="utf-8")
                return {
                    "epoch_records": [_EpochRecord()],
                    "epochs_completed": 2,
                    "steps_completed": 8,
                    "stopped_early": True,
                    "stop_reason": "early_stop",
                    "monitor_name": "primary_task_macro_f1",
                    "best_epoch": 2,
                    "best_monitor_value": 0.75,
                    "best_checkpoint": str(checkpoint_path),
                    "cache_stats": {},
                }

            def fake_test(**kwargs):
                self.assertEqual(kwargs["splits"], ["test"])
                self.assertEqual(kwargs["checkpoint_path"].name, "checkpoint_best.pt")
                captured_model_name.append(
                    resolve_encoder_config("content", "PHEME", "full").model_name("full")
                )
                result = {"sample_count": 5, "macro_f1": 0.7, "accuracy": 0.8}
                write_json(kwargs["output_path"], result)
                return result

            with (
                patch.object(pipeline, "_now_dirstamp", return_value="standard_run"),
                patch.object(pipeline, "resolve_fusion_config", return_value=_FakeConfig()),
                patch.object(pipeline, "seed_everything"),
                patch.object(pipeline, "ensure_model_inputs", return_value=[]),
                patch.object(
                    pipeline,
                    "build_train_loader",
                    return_value=(
                        train_loader,
                        10,
                        ["train"],
                        feature_dims,
                        {},
                        primary_task_spec,
                        {0: 5, 1: 5},
                    ),
                ) as build_train,
                patch.object(
                    pipeline,
                    "build_eval_loader",
                    return_value=(validation_loader, 4, ["val"], feature_dims, {}, primary_task_spec),
                ) as build_validation,
                patch.object(pipeline, "UMCLFiveViewModel", return_value=_FakeModel()),
                patch.object(pipeline, "build_hierarchical_optimizer", return_value=object()),
                patch.object(pipeline.torch.optim.lr_scheduler, "ReduceLROnPlateau", return_value=object()),
                patch.object(pipeline, "run_training", side_effect=fake_training) as run_training,
                patch.object(pipeline, "run_test", side_effect=fake_test) as run_test,
            ):
                summary = pipeline.run_pipeline(args)

            self.assertEqual(build_train.call_args.kwargs["splits"], ["train"])
            self.assertEqual(build_validation.call_args.kwargs["splits"], ["val"])
            self.assertEqual(run_training.call_count, 1)
            self.assertEqual(run_test.call_count, 1)
            self.assertEqual(summary["training"]["best_epoch"], 2)
            self.assertEqual(captured_model_name, [args.pretrained_model])

            run_dir = root / "runs" / "PHEME" / "seed_7" / "standard_run"
            expected_files = {
                "run_config.json",
                "epoch_log.jsonl",
                "validation_log.csv",
                "checkpoint_best.pt",
                "test_summary.json",
                "summary.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in run_dir.iterdir()}))
            self.assertFalse((run_dir / "eval_test_report.csv").exists())
            self.assertFalse((root / "result_umcl.csv").exists())


if __name__ == "__main__":
    unittest.main()
