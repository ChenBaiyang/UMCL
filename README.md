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

- Supported datasets: `RumourEval`, `PHEME`, `Twitter15`, `Twitter16`, `Weibo`
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
python run_umcl_pipeline.py --dataset_id RumourEval
python run_umcl_pipeline.py --dataset_id RumourEval --seed 13
python run_umcl_pipeline.py --dataset_id RumourEval --pretrained_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Generate the five legacy `train_umcl.py` validation-best checkpoints used by
the complete-only experiment manifest:

```bash
python3 run_umcl_seed_sweep.py --dataset_id PHEME
python3 run_umcl_seed_sweep.py --dataset_id Weibo
```

The sweep runs seeds 2, 4, 8, 16, and 32 with `--checkpoint true`. After each
successful legacy candidate-epoch test it records `checkpoint_best.pt` in
`results/standard_umcl_checkpoints.csv`. Existing rows whose checkpoint still
exists are skipped, so the same command can resume an interrupted sweep.

For candidate-epoch analysis, view ablations, loss sweeps, or other legacy
experiment overrides, use `train_umcl.py` explicitly.

Run test-set evaluation:

```powershell
python test_umcl.py --dataset_id RumourEval --checkpoint runs/RumourEval/seed_13/20260528_000000_000000/checkpoint_best.pt
python test_umcl.py --dataset_id RumourEval --checkpoint runs/RumourEval/seed_13/20260528_000000_000000/checkpoint_best.pt --max_batches 10
```

`test_umcl.py` defaults to the `test` split, then falls back to `all`, `val`,
`dev`, and `train` if needed. Like training, it ensures required artifacts
before building the loader.