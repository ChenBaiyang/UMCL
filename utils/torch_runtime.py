from __future__ import annotations

import os
import random


DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# This must be configured before the first CUDA/cuBLAS context is created.
# Some entrypoints perform CUDA evaluation before they call seed_everything().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE_CONFIG)


def resolve_torch_device(device: str) -> tuple[str, str | None]:
    requested = (device or "auto").strip().lower()
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu", "torch_not_installed"

    if requested == "auto":
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            return f"cuda:{current}", None
        return "cpu", "cuda_unavailable"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", "cuda_requested_but_unavailable"
    return device, None


def configure_torch_runtime(*, tokenizer_parallelism: str, torch_num_threads: int | None) -> None:
    if tokenizer_parallelism in {"true", "false"}:
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_parallelism
    if torch_num_threads is not None:
        try:
            import torch

            torch.set_num_threads(max(1, int(torch_num_threads)))
        except ModuleNotFoundError:
            pass


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None  # type: ignore[assignment]

    import torch

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = int(seed) + int(worker_id)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ModuleNotFoundError:
        pass
    import torch

    torch.manual_seed(worker_seed)
