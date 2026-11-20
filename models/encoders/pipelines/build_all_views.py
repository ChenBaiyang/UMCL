from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build encoder artifacts for all or selected views.")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--views", default="all", help="Comma-separated subset of content,stance,propagation,evidence,source")
    parser.add_argument("--input_root", default="data/preprocessed")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--stance_device", default="auto")
    parser.add_argument("--stance_tokenizer_parallelism", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--stance_torch_num_threads", type=int, default=None)
    parser.add_argument("--stance_gc_interval_shards", type=int, default=8)
    parser.add_argument("--stance_encode_batch_size_override", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    from preprocess.preprocess_views import parse_views, preprocess_views

    args = parse_args()
    preprocess_views(
        args.dataset_id,
        "full",
        input_root=Path(args.input_root),
        data_root=Path(args.data_root),
        views=parse_views(args.views),
        stance_device=args.stance_device,
        stance_tokenizer_parallelism=args.stance_tokenizer_parallelism,
        stance_torch_num_threads=args.stance_torch_num_threads,
        stance_gc_interval_shards=args.stance_gc_interval_shards,
        stance_encode_batch_size_override=args.stance_encode_batch_size_override,
    )


if __name__ == "__main__":
    main()
