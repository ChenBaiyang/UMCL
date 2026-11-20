"""Train the internal fusion variants using the shared experimental runner."""
from __future__ import annotations

import argparse

from models.training import run_training_cli
from train_umcl import build_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    parser.description = "Compare concat, gated, and mean-posterior fusion with UMCL."
    parser.set_defaults(run_root="runs_internal_fusion")
    parser.add_argument("--fusion_variant", required=True,
                        choices=["concat", "gated", "mean_posterior"])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_training_cli(args, fusion_variant=args.fusion_variant)


if __name__ == "__main__":
    main()
