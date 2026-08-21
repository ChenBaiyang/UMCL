from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_SUFFIX = ".mmap"
MANIFEST_FILENAME = "bundle_manifest.json"


def bundle_dir_for(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.name + BUNDLE_SUFFIX)


def write_array_bundle(npz_path: Path, arrays: dict[str, Any]) -> Path:
    bundle_dir = bundle_dir_for(npz_path)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"keys": {}}
    for key, value in arrays.items():
        array = np.asarray(value)
        filename = f"{key}.npy"
        np.save(bundle_dir / filename, array, allow_pickle=bool(array.dtype == object))
        manifest["keys"][key] = {
            "filename": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "allow_pickle": bool(array.dtype == object),
        }
    (bundle_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle_dir


def read_array_bundle_manifest(npz_path: Path) -> dict[str, Any] | None:
    manifest_path = bundle_dir_for(npz_path) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))

