"""Command-line entry for evaluating a UMCL checkpoint."""
from __future__ import annotations

import argparse

from models.evaluation import run_evaluation_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True, help="Single dataset id, e.g. PHEME")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained fusion checkpoint.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for test batches, useful for quick checks.")
    parser.add_argument("--splits", default=None, help="Optional comma-separated splits. Default: test, then all, then val/dev, then train.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--output_path", default=None, help="Optional path for test summary JSON.")
    parser.add_argument("--log_path", default=None, help="Optional path for per-batch test log JSONL.")
    parser.add_argument("--efficiency", action="store_true", help="Run the standardized efficiency benchmark.")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    import json
    print(json.dumps(run_evaluation_cli(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
