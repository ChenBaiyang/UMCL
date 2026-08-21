from __future__ import annotations

import argparse
from pathlib import Path


def build_test_loader_with_fallback(*args, **kwargs):
    from test_umcl import build_test_loader_with_fallback as _impl

    return _impl(*args, **kwargs)


def run_test_from_model(*args, **kwargs):
    from test_umcl import run_test_from_model as _impl

    return _impl(*args, **kwargs)


def run_test(*args, **kwargs):
    from test_umcl import run_test as _impl

    return _impl(*args, **kwargs)


def run_evaluation_cli(args: argparse.Namespace) -> dict:
    from test_umcl import resolve_amp, resolve_device

    dataset_id = args.dataset_id.strip()
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    device = resolve_device(args.device)
    use_amp = resolve_amp(args.amp, device)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()] if args.splits else None
    return run_test(
        dataset_id=dataset_id,
        checkpoint_path=checkpoint_path,
        mode="full",
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        disable_persistent_workers=args.disable_persistent_workers,
        disable_pin_memory=args.disable_pin_memory,
        max_batches=args.max_batches,
        splits=splits,
        device=device,
        use_amp=use_amp,
        output_path=None,
        log_path=None,
    )
