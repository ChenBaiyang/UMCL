#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"


if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset1> [dataset2] ..."
    echo "Example: $0 Twitter16 PHEME"
    exit 1
fi


for DATASET in "$@"; do
    for SEED in 2 4 8 16 32; do
        echo "Running dataset: $DATASET, seed: $SEED"
        python train_umcl.py --dataset_id "$DATASET" --seed "$SEED"
    done
done

echo "Results" | msmtp --account=163mail  farstars@qq.com < content_pool.txt