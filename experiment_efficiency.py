"""UMCL experiment: efficiency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = (
    "BiGCN", "PLAN", "MST", "BCFN", "RRD-N", "UMCL",
    "TD-RvNN", "CT-RvNN", "EBGCN", "GMVCN", "AD-GSCL",
)
DEFAULT_SEEDS = (2, 4, 8, 16, 32)


def _current_rss_bytes() -> int:
    """Return Linux process RSS without adding a psutil dependency."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        raise RuntimeError("CPU memory benchmarking requires Linux /proc/self/status.")
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was not found in /proc/self/status.")


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.001) -> None:
        self.interval_seconds = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_RssSampler":
        self.peak = _current_rss_bytes()

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak = max(self.peak, _current_rss_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, _current_rss_bytes())


def _unique_parameters(modules: Sequence[torch.nn.Module]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result: list[torch.nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> float:
    scores: list[float] = []
    for class_id in range(num_classes):
        truth = labels == class_id
        predicted = predictions == class_id
        tp = int((truth & predicted).sum().item())
        fp = int((~truth & predicted).sum().item())
        fn = int((truth & ~predicted).sum().item())
        denominator = 2 * tp + fp + fn
        scores.append((2.0 * tp / denominator) if denominator else 0.0)
    return float(sum(scores) / num_classes)


def benchmark_model(
    *,
    method: str,
    modules: Sequence[torch.nn.Module],
    loader: Iterable[Any],
    forward_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
    batch_size: int,
    warmup_runs: int,
    benchmark_runs: int,
    dataset: str = "PHEME",
    seed: int | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Benchmark model-ready batches; loader I/O and collation are not timed."""
    if warmup_runs < 0 or benchmark_runs <= 0:
        raise ValueError("warmup_runs must be >= 0 and benchmark_runs must be > 0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested, but torch.cuda.is_available() is false.")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported benchmark device: {device}")

    for module in modules:
        module.eval()
    parameters = _unique_parameters(modules)
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)

    def one_pass(measured: bool) -> dict[str, Any]:
        labels_parts: list[torch.Tensor] = []
        prediction_parts: list[torch.Tensor] = []
        elapsed = 0.0
        sample_count = 0
        rss_sampler = _RssSampler() if measured and device.type == "cpu" else None
        if measured and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        context = rss_sampler if rss_sampler is not None else _NullContext()
        with context, torch.inference_mode():
            for batch in loader:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                probabilities, labels = forward_batch(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - started
                if probabilities.ndim != 2:
                    raise ValueError(f"Expected [batch, classes] probabilities, got {tuple(probabilities.shape)}")
                labels = labels.view(-1)
                if probabilities.shape[0] != labels.shape[0]:
                    raise ValueError("Probability and label batch sizes differ.")
                sample_count += int(labels.numel())
                prediction_parts.append(probabilities.argmax(dim=-1).detach().cpu())
                labels_parts.append(labels.detach().cpu())
        if sample_count == 0:
            raise ValueError("Benchmark loader produced zero events.")
        all_labels = torch.cat(labels_parts)
        all_predictions = torch.cat(prediction_parts)
        if measured and device.type == "cuda":
            peak_bytes = int(torch.cuda.max_memory_allocated(device))
            memory_metric = "cuda_peak_allocated"
        elif measured and rss_sampler is not None:
            peak_bytes = int(rss_sampler.peak)
            memory_metric = "process_peak_rss"
        else:
            peak_bytes = 0
            memory_metric = "warmup_not_recorded"
        return {
            "sample_count": sample_count,
            "elapsed_seconds": elapsed,
            "inference_ms_per_event": elapsed * 1000.0 / sample_count,
            "peak_memory_bytes": peak_bytes,
            "peak_memory_gb": peak_bytes / (1024.0**3),
            "memory_metric": memory_metric,
            "macro_f1": _macro_f1(all_labels, all_predictions, num_classes),
        }

    def reset_inference_rng() -> None:
        # RRD-N's BayesianLinear samples weights even in eval mode. Replaying the
        # same RNG stream keeps full-test predictions comparable across timing
        # repeats without replacing its stochastic inference implementation.
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    for _ in range(warmup_runs):
        reset_inference_rng()
        one_pass(measured=False)
    repeats = []
    for _ in range(benchmark_runs):
        reset_inference_rng()
        repeats.append(one_pass(measured=True))
    counts = {int(item["sample_count"]) for item in repeats}
    f1_values = [float(item["macro_f1"]) for item in repeats]
    if len(counts) != 1:
        raise RuntimeError("Repeated evaluations produced inconsistent sample counts.")
    predictions_deterministic = max(f1_values) - min(f1_values) <= 1e-12
    if not predictions_deterministic:
        raise RuntimeError(
            "Repeated evaluations produced inconsistent Macro-F1 values even after "
            f"resetting the inference RNG ({min(f1_values):.6f}..{max(f1_values):.6f})."
        )
    time_values = [float(item["inference_ms_per_event"]) for item in repeats]
    memory_values = [float(item["peak_memory_gb"]) for item in repeats]
    return {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "checkpoint": checkpoint,
        "device": str(device),
        "precision": "fp32",
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "sample_count": counts.pop(),
        "macro_f1": statistics.fmean(f1_values),
        "macro_f1_std": statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0,
        "predictions_deterministic": predictions_deterministic,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "inference_ms_per_event_mean": statistics.fmean(time_values),
        "inference_ms_per_event_std": statistics.stdev(time_values) if len(time_values) > 1 else 0.0,
        "peak_memory_gb_mean": statistics.fmean(memory_values),
        "peak_memory_gb_std": statistics.stdev(memory_values) if len(memory_values) > 1 else 0.0,
        "memory_metric": repeats[0]["memory_metric"],
        "repeats": repeats,
    }


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def write_benchmark_json(path: str | Path, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class MethodSpec:
    directory: str | None
    preprocess_script: str | None
    train_script: str
    batch_argument: str
    data_directory: str | None


METHOD_SPECS = {
    "BiGCN": MethodSpec("baselines/BiGCN", "preprocess_bigcn.py", "train_bigcn.py", "--batchsize", "data/bigcn/pheme"),
    "PLAN": MethodSpec("baselines/PLAN", "preprocess_plan.py", "train_plan.py", "--batch_size", "data/plan/pheme"),
    "MST": MethodSpec("baselines/MST", "preprocess_mst.py", "train_mst.py", "--batch", "data/mst/pheme"),
    "BCFN": MethodSpec("baselines/BCFN", "preprocess_bcfn.py", "train_bcfn.py", "--batch_size", "data/bcfn/pheme"),
    "RRD-N": MethodSpec("baselines/RRD-N", "preprocess_rrd_n.py", "train_rrd_n.py", "--batchsize", "data/rrd_n/pheme"),
    "UMCL": MethodSpec(None, None, "train_umcl.py", "--batch_size", None),
    "TD-RvNN": MethodSpec("baselines/TD_RvNN", "preprocess_td_rvnn.py", "train_td_rvnn.py", "--batch_size", "data/td_rvnn/pheme"),
    "CT-RvNN": MethodSpec("baselines/CT-RvNN", "preprocess.py", "train.py", "--batch_size", None),
    "EBGCN": MethodSpec("baselines/EBGCN", "preprocess_ebgcn.py", "train_ebgcn.py", "--batchsize", "data/ebgcn/pheme"),
    "GMVCN": MethodSpec("baselines/GMVCN", "preprocess.py", "train_gmvcn.py", "--batch_size", "data/gmvcn/pheme"),
    "AD-GSCL": MethodSpec("baselines/AD-GSCL", "preprocess_ad_gscl.py", "train_ad_gscl.py", "--batch_size", "data/ad_gscl/pheme"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark PHEME efficiency baselines on Ubuntu/CUDA.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=5)
    parser.add_argument("--stage_timeout_seconds", type=float, default=3600.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output_dir", default="results/computational_efficiency/pheme")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.stage_timeout_seconds <= 0:
        parser.error("--stage_timeout_seconds must be greater than zero")
    return args


class CommandTimeoutError(RuntimeError):
    pass


def _terminate_process_tree(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        os.killpg(process.pid, signal.SIGKILL)
    else:
        if process.poll() is not None:
            return
        process.kill()
    if process.poll() is None:
        process.wait()


def _run_command(
    command: list[str], *, cwd: Path, log_path: Path, dry_run: bool,
    timeout_seconds: float | None = None,
) -> None:
    rendered = " ".join(command)
    print(f"[{cwd}] {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=os.name == "posix",
        )
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            raise CommandTimeoutError(
                f"Command exceeded {timeout_seconds:g} seconds; see {log_path}"
            ) from exc
        except KeyboardInterrupt:
            _terminate_process_tree(process)
            raise
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")


def _probe_runtime(python_executable: str, cuda_index: int) -> dict[str, Any]:
    code = (
        "import json,platform,sys,torch; "
        f"i={cuda_index}; "
        "print(json.dumps({'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,"
        "'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'gpu':torch.cuda.get_device_name(i) if torch.cuda.is_available() and i < torch.cuda.device_count() else None}))"
    )
    completed = subprocess.run(
        [python_executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not inspect --python runtime: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _external_commands(method: str, seed: int, args: argparse.Namespace, seed_dir: Path) -> tuple[list[str], list[str]]:
    spec = METHOD_SPECS[method]
    checkpoint = seed_dir / "checkpoint_best.pt"
    result = seed_dir / "benchmark.json"
    common = ["--dataset", "PHEME", "--seed", str(seed), spec.batch_argument, str(args.batch_size)]
    if args.device == "cpu":
        common.append("--no_cuda")
    else:
        if method in {"BiGCN", "RRD-N", "EBGCN"}:
            common += ["--num_cuda", str(args.cuda_index)]
        elif method in {"PLAN", "MST", "BCFN", "TD-RvNN", "CT-RvNN", "GMVCN", "AD-GSCL"}:
            common += ["--cuda", str(args.cuda_index)]
    if method == "CT-RvNN":
        common += ["--depth", "7", "--node_feature_type", "tfidf"]
    train = [args.python, spec.train_script, *common, "--task", "train", "--checkpoint_path", str(checkpoint)]
    efficiency = [
        args.python,
        spec.train_script,
        *common,
        "--task",
        "efficiency",
        "--checkpoint_path",
        str(checkpoint),
        "--warmup_runs",
        str(args.warmup_runs),
        "--benchmark_runs",
        str(args.benchmark_runs),
        "--efficiency_output",
        str(result),
    ]
    return train, efficiency


def _find_umcl_checkpoint(run_root: Path) -> Path:
    checkpoints = sorted(run_root.rglob("checkpoint_best.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"UMCL training did not create a checkpoint under {run_root}")
    return checkpoints[-1]


def _preprocessed_available(method: str) -> bool:
    if method == "CT-RvNN":
        root = PROJECT_ROOT / "baselines" / "CT-RvNN"
        return all((root / f"pheme_trees_{split}_7.pkl").is_file() for split in ("train", "val", "test"))
    root = PROJECT_ROOT / str(METHOD_SPECS[method].data_directory)
    if method in {"BiGCN", "RRD-N"}:
        return all(any((root / split).glob("*.npz")) for split in ("train_npz", "val_npz", "test_npz"))
    if method == "PLAN":
        return all((root / name).exists() for name in ("metadata.json", "vocab.json", "train_npz", "eval_npz", "test_npz"))
    if method == "MST":
        return (root / "pheme_temporal_data" / "metadata.json").is_file()
    if method == "BCFN":
        return (root / "bcfn_cache.pkl").is_file()
    if method == "TD-RvNN":
        return (root / "td_rvnn_tfidf.pkl").is_file()
    if method == "EBGCN":
        return all(any((root / split).glob("*.npz")) for split in ("train_npz", "val_npz", "test_npz"))
    if method == "GMVCN":
        return all(any((root / split).glob("*.npz")) for split in ("train_tfidfNPZ", "eval_tfidfNPZ", "test_tfidfNPZ"))
    if method == "AD-GSCL":
        return (root / "ad_gscl_tfidf.pkl").is_file()
    return True


def _umcl_commands(seed: int, args: argparse.Namespace, seed_dir: Path) -> tuple[list[str], Callable[[], list[str]]]:
    run_root = seed_dir / "runs"
    device = "cpu" if args.device == "cpu" else "cuda"
    train = [
        args.python,
        str(PROJECT_ROOT / "train_umcl.py"),
        "--dataset_id",
        "PHEME",
        "--seed",
        str(seed),
        "--batch_size",
        str(args.batch_size),
        "--run_root",
        str(run_root),
        "--checkpoint",
        "true",
        "--device",
        device,
        "--cuda_index",
        str(args.cuda_index),
        "--amp",
        "false",
    ]

    def efficiency() -> list[str]:
        return [
            args.python,
            str(PROJECT_ROOT / "test_umcl.py"),
            "--dataset_id",
            "PHEME",
            "--checkpoint",
            str(_find_umcl_checkpoint(run_root)),
            "--batch_size",
            str(args.batch_size),
            "--device",
            device,
            "--cuda_index",
            str(args.cuda_index),
            "--amp",
            "false",
            "--efficiency",
            "--warmup_runs",
            str(args.warmup_runs),
            "--benchmark_runs",
            str(args.benchmark_runs),
            "--output_path",
            str(seed_dir / "benchmark.json"),
        ]

    return train, efficiency


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _normalized_device(value: Any) -> str:
    device = str(value).strip().lower()
    return "cuda:0" if device == "cuda" else device


def _result_matches(
    path: Path, method: str, seed: int, args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read valid JSON: {exc}"
    expected_device = "cpu" if args.device == "cpu" else f"cuda:{args.cuda_index}"
    try:
        comparisons = {
            "method": (str(result.get("method", "")).strip().lower(), method.lower()),
            "seed": (int(result["seed"]) if result.get("seed") is not None else None, int(seed)),
            "device": (_normalized_device(result.get("device")), _normalized_device(expected_device)),
            "precision": (str(result.get("precision", "")).lower(), "fp32"),
            "batch_size": (int(result.get("batch_size", -1)), int(args.batch_size)),
            "warmup_runs": (int(result.get("warmup_runs", -1)), int(args.warmup_runs)),
            "benchmark_runs": (int(result.get("benchmark_runs", -1)), int(args.benchmark_runs)),
        }
    except (TypeError, ValueError) as exc:
        return None, f"invalid configuration field: {exc}"
    required_metrics = (
        "macro_f1", "total_parameters", "trainable_parameters",
        "inference_ms_per_event_mean", "peak_memory_gb_mean",
    )
    missing = [key for key in required_metrics if result.get(key) is None]
    mismatches = [f"{key}={actual!r} (expected {expected!r})" for key, (actual, expected) in comparisons.items() if actual != expected]
    if missing:
        mismatches.append(f"missing metrics: {', '.join(missing)}")
    return (result, None) if not mismatches else (None, "; ".join(mismatches))


def _write_outputs(output_dir: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_fields = [
        "method", "seed", "macro_f1", "macro_f1_std", "predictions_deterministic",
        "total_parameters", "trainable_parameters", "sample_count",
        "inference_ms_per_event_mean", "inference_ms_per_event_std", "peak_memory_gb_mean",
        "peak_memory_gb_std", "memory_metric", "device", "precision", "batch_size",
    ]
    with (output_dir / "efficiency_per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    summaries: list[dict[str, Any]] = []
    failed_methods = {str(failure["method"]) for failure in failures if failure.get("method")}
    for method in args.methods:
        if method in failed_methods:
            continue
        rows = [row for row in results if row["method"] == method]
        if not rows:
            continue
        totals = {int(row["total_parameters"]) for row in rows}
        trainables = {int(row["trainable_parameters"]) for row in rows}
        if len(totals) != 1 or len(trainables) != 1:
            raise RuntimeError(f"Parameter counts changed across seeds for {method}.")
        f1_mean, f1_std = _mean_std([float(row["macro_f1"]) for row in rows])
        time_mean, time_std = _mean_std([float(row["inference_ms_per_event_mean"]) for row in rows])
        memory_mean, memory_std = _mean_std([float(row["peak_memory_gb_mean"]) for row in rows])
        summaries.append({
            "method": method,
            "seed_count": len(rows),
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "total_parameters": totals.pop(),
            "trainable_parameters": trainables.pop(),
            "inference_ms_per_event_mean": time_mean,
            "inference_ms_per_event_std": time_std,
            "peak_memory_gb_mean": memory_mean,
            "peak_memory_gb_std": memory_std,
            "memory_metric": rows[0]["memory_metric"],
        })
    summary_fields = list(summaries[0]) if summaries else ["method"]
    with (output_dir / "efficiency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    latex_lines = [
        f'{row["method"]} & {row["macro_f1_mean"]:.4f} & '
        f'{row["total_parameters"]}/{row["trainable_parameters"]} & '
        f'{row["inference_ms_per_event_mean"]:.3f} & {row["peak_memory_gb_mean"]:.3f} \\\\'
        for row in summaries
    ]
    (output_dir / "end_to_end_efficiency_rows.tex").write_text("\n".join(latex_lines) + ("\n" if latex_lines else ""), encoding="utf-8")
    metadata = {
        "schema_version": 2,
        "status": "failed" if failures else "complete",
        "arguments": vars(args),
        "environment": getattr(args, "runtime_info", {
            "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
            "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": None,
        }),
        "results": results,
        "failures": failures,
        "abandoned_methods": sorted({str(item["method"]) for item in failures if item.get("method")}),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.python = str(Path(args.python).expanduser().resolve())
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not args.dry_run:
        args.runtime_info = _probe_runtime(args.python, args.cuda_index)
        if args.device == "cuda" and (
            not args.runtime_info["cuda_available"] or args.cuda_index >= args.runtime_info["cuda_device_count"]
        ):
            raise RuntimeError("The selected --python runtime cannot access the requested CUDA device. Use --device cpu only for debugging.")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method in args.methods:
        spec = METHOD_SPECS[method]
        pending_seeds: list[int] = []
        for seed in args.seeds:
            result_path = output_dir / method / f"seed_{seed}" / "benchmark.json"
            if result_path.exists() and not args.overwrite and not args.dry_run:
                cached, mismatch_reason = _result_matches(result_path, method, seed, args)
                if cached is not None:
                    results.append(cached)
                    print(f"[skip completed] method={method} seed={seed} result={result_path}", flush=True)
                    continue
                print(
                    f"[rerun] method={method} seed={seed}: existing benchmark is incompatible: {mismatch_reason}",
                    flush=True,
                )
            pending_seeds.append(seed)

        if not pending_seeds and not args.force_preprocess:
            print(f"[skip method] method={method}: all requested seeds are complete", flush=True)
            continue

        if spec.preprocess_script is not None:
            method_cwd = PROJECT_ROOT / str(spec.directory)
            if args.force_preprocess or not _preprocessed_available(method):
                command = [args.python, spec.preprocess_script, "--dataset", "PHEME"]
                if method == "CT-RvNN":
                    command += ["--height", "7", "--node_feature_type", "tfidf"]
                try:
                    _run_command(command, cwd=method_cwd, log_path=output_dir / method / "preprocess.log", dry_run=args.dry_run)
                except Exception as exc:
                    failures.append({
                        "method": method, "stage": "preprocess", "error": str(exc),
                        "timed_out": isinstance(exc, CommandTimeoutError),
                        "abandoned_seeds": pending_seeds,
                    })
                    if args.fail_fast:
                        _write_outputs(output_dir, results, failures, args)
                        return 1
                    continue
        for seed_index, seed in enumerate(pending_seeds):
            seed_dir = output_dir / method / f"seed_{seed}"
            result_path = seed_dir / "benchmark.json"
            current_stage = "train"
            try:
                if method == "UMCL":
                    train_command, efficiency_factory = _umcl_commands(seed, args, seed_dir)
                    existing_umcl_checkpoints = list((seed_dir / "runs").rglob("checkpoint_best.pt"))
                    if existing_umcl_checkpoints and not args.overwrite and not args.dry_run:
                        print(f"[skip training] method={method} seed={seed}: checkpoint exists", flush=True)
                    else:
                        _run_command(
                            train_command, cwd=PROJECT_ROOT, log_path=seed_dir / "train.log",
                            dry_run=args.dry_run, timeout_seconds=args.stage_timeout_seconds,
                        )
                    current_stage = "efficiency"
                    if args.dry_run:
                        print(f"[{PROJECT_ROOT}] <test_umcl.py command resolved after checkpoint creation>")
                    else:
                        _run_command(
                            efficiency_factory(), cwd=PROJECT_ROOT, log_path=seed_dir / "efficiency.log",
                            dry_run=False, timeout_seconds=args.stage_timeout_seconds,
                        )
                else:
                    train_command, efficiency_command = _external_commands(method, seed, args, seed_dir)
                    method_cwd = PROJECT_ROOT / str(spec.directory)
                    checkpoint_path = seed_dir / "checkpoint_best.pt"
                    if checkpoint_path.is_file() and not args.overwrite and not args.dry_run:
                        print(
                            f"[skip training] method={method} seed={seed}: checkpoint={checkpoint_path}",
                            flush=True,
                        )
                    else:
                        _run_command(
                            train_command, cwd=method_cwd, log_path=seed_dir / "train.log",
                            dry_run=args.dry_run, timeout_seconds=args.stage_timeout_seconds,
                        )
                    current_stage = "efficiency"
                    _run_command(
                        efficiency_command, cwd=method_cwd, log_path=seed_dir / "efficiency.log",
                        dry_run=args.dry_run, timeout_seconds=args.stage_timeout_seconds,
                    )
                if not args.dry_run:
                    current_stage = "result"
                    results.append(json.loads(result_path.read_text(encoding="utf-8")))
            except Exception as exc:
                abandoned_seeds = pending_seeds[seed_index + 1:]
                failures.append({
                    "method": method, "seed": seed, "stage": current_stage, "error": str(exc),
                    "timed_out": isinstance(exc, CommandTimeoutError),
                    "timeout_seconds": args.stage_timeout_seconds if isinstance(exc, CommandTimeoutError) else None,
                    "abandoned_seeds": abandoned_seeds,
                })
                if args.fail_fast:
                    _write_outputs(output_dir, results, failures, args)
                    return 1
                print(
                    f"[abandon method] method={method} failed at seed={seed} stage={current_stage}; "
                    f"skipping seeds={abandoned_seeds}",
                    flush=True,
                )
                break
    if not args.dry_run:
        _write_outputs(output_dir, results, failures, args)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
