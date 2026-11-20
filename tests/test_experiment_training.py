from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import experiment_fusion_comparison as comparison
import models.training as training
import train_umcl
from utils.primary_label_space import resolve_primary_task_spec


@pytest.mark.parametrize("variant", ["poe", "concat", "gated", "mean_posterior"])
def test_shared_runner_preserves_checkpoint_policy_and_outputs(tmp_path, monkeypatch, variant):
    argv = ["--dataset_id", "PHEME", "--device", "cpu", "--max_epoch", "3",
            "--candidate_epochs", "2", "--run_root", str(tmp_path)]
    args = train_umcl.parse_args(argv) if variant == "poe" else comparison.parse_args([
        *argv, "--fusion_variant", variant])
    spec = resolve_primary_task_spec("PHEME")
    dims = {"content_surface": 3}
    model = Mock()
    model.to.return_value = model
    monkeypatch.setattr(training, "ensure_model_inputs", lambda **kwargs: [])
    monkeypatch.setattr(training, "create_fusion_model", Mock(return_value=model))
    monkeypatch.setattr(training, "build_hierarchical_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training.torch.optim.lr_scheduler, "ReduceLROnPlateau", lambda *args, **kwargs: object())
    monkeypatch.setattr(training, "build_train_loader", lambda **kwargs: (object(), 10, ["train"], dims, {}, spec, {0: 5}))
    monkeypatch.setattr(training, "build_eval_loader", lambda **kwargs: (object(), 4, ["val"], dims, {}, spec))
    monkeypatch.setattr(training, "build_test_loader_with_fallback", lambda **kwargs: (object(), 4, ["test"], dims, {}, spec))
    metric = {"macro_f1": .7, "accuracy": .8}
    monkeypatch.setattr(training, "run_test_from_model", lambda **kwargs: {
        "metric_name": spec["metric_name"], "metrics": {spec["metric_name"]: metric},
        **metric, "per_class_f1": {name: .7 for name in spec["class_names"]}})
    appended = Mock()
    monkeypatch.setattr(train_umcl, "_append_result_umcl_row", appended)
    monkeypatch.setattr(train_umcl, "_build_summary", lambda **kwargs: {
        "run": {}, "hyperparameters": {},
        "training": {"best_epoch": kwargs["training_result"]["best_epoch"]}})
    observed = {}

    def fake_training(**kwargs):
        observed.update(kwargs)
        records = []
        for epoch, monitor in [(1, .9), (2, .6), (3, .7)]:
            best_epoch = epoch if variant == "poe" else 1
            record = SimpleNamespace(epoch=epoch, head_lr=.001, is_best=True, best_epoch=best_epoch,
                                     monitor_name="macro_f1", monitor_value=monitor,
                                     best_monitor_value=monitor, early_stop_counter=0,
                                     train_losses={"task_primary_loss": .5})
            event = {"epoch": epoch, "model": model, "epoch_record": record,
                     "checkpoint_path": None, "checkpoint_saved": False, "monitor_value": monitor}
            kwargs["on_new_best"](event)
            kwargs["on_epoch_end"](event)
            records.append(record)
        return {"last_step_record": object(), "epoch_records": records,
                "best_epoch": 3 if variant == "poe" else 1}

    monkeypatch.setattr(training, "run_training", fake_training)
    training.run_training_cli(args, fusion_variant=variant)
    assert observed["enable_early_stop"] is False
    assert observed["config"].fusion_variant == variant
    if variant == "poe":
        assert observed["best_selection_start_epoch"] == 2
        assert observed["best_selection_min_delta"] == 0.
        assert appended.call_args.args[0].name == "result_umcl.csv"
    else:
        assert "best_selection_start_epoch" not in observed
        assert appended.call_args.args[0].name == "result_internal_fusion.csv"
        assert appended.call_args.args[2]["fusion_variant"] == variant
        assert observed["checkpoint_dir"].relative_to(tmp_path).parts[:2] == (variant, "PHEME")
    summary = json.loads((observed["checkpoint_dir"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["testing"]["best_epoch"] == 3
    assert summary["training"]["best_epoch"] == (3 if variant == "poe" else 1)
