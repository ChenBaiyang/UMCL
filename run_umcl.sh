#!/bin/bash

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset1> [dataset2] ..."
    echo "Example: $0 Twitter16 PHEME"
    exit 1
fi


for DATASET in "$@"; do
    for SEED in 2 4 8 16 32; do
        echo "Running dataset: $DATASET, seed: $SEED"
        "$PYTHON_BIN" run_umcl_pipeline.py --dataset_id "$DATASET" --seed "$SEED"
    done
done

