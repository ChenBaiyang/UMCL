# UMCL

UMCL is a unified multi-view credibility learning project with four active stages:

1. `raw downloads -> preprocessed UnifiedEvent`
2. `preprocessed UnifiedEvent -> view artifacts`
3. `view artifacts -> encoder + PoE fusion training`
4. `checkpoint + view artifacts -> metrics`

The formal stage entrypoints are now:

1. Event preprocessing: `preprocess/preprocess_events.py`
2. View preprocessing: `preprocess/preprocess_views.py`
3. Standard train/validation/test pipeline: `run_umcl_pipeline.py`
4. Evaluation: `test_umcl.py -> models/evaluation.py`

`train_umcl.py` remains available as the legacy experimental training entrypoint.

## Configuration

Runtime configuration is resolved from:

- `config/config.json`
- `dataset_id`

UMCL now supports only the `full` runtime configuration.

## Current Stack

- Supported datasets: `PHEME`, `Twitter15`, `Twitter16`, `Weibo`
- Supported views: `content`, `stance`, `propagation`, `evidence`, `source`
- Main pipeline entry: `run_umcl_pipeline.py`
- Legacy experimental training entry: `train_umcl.py`
- Main test entry: `test_umcl.py`

## Standard Pipeline Runtime

`run_umcl_pipeline.py` exposes only the core reproducibility and runtime options:

- `--seed`
- `--pretrained_model`
- `--device auto|cpu|cuda`
- `--data_root`, `--run_root`

Model, optimizer, scheduler, batch-size, active-view, epoch, and early-stop
settings are resolved from `config/config.json`. The standard pipeline:

- computes the minimal required artifact set from the effective `active_views`
- ensures the content event spine in `data/content_view/<dataset>/<mode>/event_index.npz`
- ensures only the view artifacts required by the run
- reuses cached artifacts when the saved fingerprint still matches the current configuration
- rebuilds only the affected artifact when the fingerprint or event-spine hash changes
- trains on `train` and selects the best checkpoint on `val`
- enables configured early stopping and always saves `checkpoint_best.pt`
- reloads that checkpoint and evaluates exactly once on `test`

Artifact cache metadata is stored as `artifact_metadata.json` inside each
view-mode artifact directory.

Validation runs at the end of every epoch. Best-epoch selection uses the
configured `monitor_name`, which defaults to `macro_f1`. The standard pipeline
does not run candidate-epoch test analysis or append to `result_umcl.csv`.

## Quick Start

Run the standard UMCL pipeline on one dataset:

```powershell
python run_umcl_pipeline.py --dataset_id PHEME
python run_umcl_pipeline.py --dataset_id PHEME --seed 13
python run_umcl_pipeline.py --dataset_id PHEME --pretrained_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Generate the five legacy `train_umcl.py` candidate-epoch-best checkpoints used by
the complete-only experiment manifest:

```bash
python3 experiment_seed_sweep.py --dataset_id PHEME
python3 experiment_seed_sweep.py --dataset_id Weibo
```

The sweep runs seeds 2, 4, 8, 16, and 32 with `--checkpoint true`. After each
successful legacy candidate-epoch test it records `checkpoint_best.pt` in
`results/standard_umcl_checkpoints.csv`. The legacy path still validates every
epoch so the scheduler trajectory is unchanged, but only the final
`candidate_epochs` window can update the best marker or checkpoint. Existing
rows are skipped only when their checkpoint contains valid candidate-selection
metadata; old full-run validation-best rows are retrained and replaced.

For candidate-epoch analysis, view ablations, loss sweeps, or other legacy
experiment overrides, use `train_umcl.py` explicitly.

Run test-set evaluation:

```powershell
python test_umcl.py --dataset_id PHEME --checkpoint runs/PHEME/seed_13/<run>/checkpoint_best.pt
python test_umcl.py --dataset_id PHEME --checkpoint runs/PHEME/seed_13/<run>/checkpoint_best.pt --max_batches 10
```

`test_umcl.py` defaults to the `test` split, then falls back to `all`, `val`,
`dev`, and `train` if needed. Like training, it ensures required artifacts
before building the loader.

## Experiment entrypoints

Experiment scripts use the `experiment_<purpose>.py` convention. Previous
filenames have been removed; update local commands using this table.

| Purpose | Current entrypoint | Previous entrypoint |
| --- | --- | --- |
| Fusion comparison | `experiment_fusion_comparison.py` | `train_internal_fusion.py` |
| Missing-view robustness | `experiment_missing_views.py` | `train_missing_view_robustness.py` |
| Mixed-subset inference | `experiment_mixed_subsets.py` | `train_unified_mixed_subset.py` |
| Five-seed checkpoint sweep | `experiment_seed_sweep.py` | `run_umcl_seed_sweep.py` |
| End-to-end efficiency | `experiment_efficiency.py` | `evaluate_computational_efficiency.py` |
| Post-encoder fusion efficiency | `experiment_fusion_efficiency.py` | `evaluate_fusion_efficiency.py` |

The shell launchers are `run_experiment_fusion_comparison.sh` and
`run_experiment_mixed_subsets.sh`. Use `--help` on each Python entrypoint for
its options. Data locations, result filenames, and experiment protocols remain
unchanged.

`train_umcl.py` and fusion comparison share `models/training.py`. The UMCL
checkpoint remains candidate-window selected; fusion comparison retains its
existing full-run checkpoint selection and candidate-window test reporting.
The standard `run_umcl_pipeline.py` continues to select on validation and test
the saved checkpoint once. Evaluation implementations live in
`models/evaluation.py`; `test_umcl.py` owns only the command-line interface.

Runtime helpers are shared in `utils/runtime.py`. Validation remains at dataset,
shape, label, split, numerical, and checkpoint boundaries. Repeated internal
type/schema checks and checkpoint reloads used solely to verify metadata have
been removed.

## Historical research materials

Paper drafts, research instructions, diagnostic reports, result tables, and
workbooks are retained as historical records. They can mention retired datasets,
module paths, and experimental designs; use this README and the current runtime
configuration for supported commands and datasets.
