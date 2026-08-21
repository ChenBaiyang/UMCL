from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.event_pipeline.dataset_ingest import available_dataset_ids
from preprocess.event_pipeline.raw_archive import build_download_archive, prune_download_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create zip archives for raw downloads.")
    parser.add_argument("--dataset_id", required=True, help="Dataset id or 'all'.")
    parser.add_argument("--downloads_root", default="data/downloads")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prune", action="store_true", help="Remove the source directory after archiving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloads_root = Path(args.downloads_root)
    dataset_ids = available_dataset_ids() if args.dataset_id == "all" else [args.dataset_id]
    for dataset_id in dataset_ids:
        archive_path = build_download_archive(dataset_id, downloads_root, overwrite=args.overwrite)
        print(f"{dataset_id}: {archive_path}")
        if args.prune:
            prune_download_source(dataset_id, downloads_root)


if __name__ == "__main__":
    main()
