from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.array_bundle import bundle_dir_for, read_array_bundle_manifest


class NpzShardCache:
    def __init__(self, max_open_shards: int = 16) -> None:
        self.max_open_shards = max(1, int(max_open_shards))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._bundle_manifests: dict[Path, dict[str, Any] | None] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _manifest_for(self, path: Path) -> dict[str, Any] | None:
        if path not in self._bundle_manifests:
            self._bundle_manifests[path] = read_array_bundle_manifest(path)
        return self._bundle_manifests[path]

    def _bundle_path_for_key(self, path: Path, key: str) -> Path | None:
        manifest = self._manifest_for(path)
        if manifest is None:
            return None
        key_meta = (manifest.get("keys") or {}).get(key)
        if not key_meta:
            return None
        return bundle_dir_for(path) / str(key_meta["filename"])

    def _disable_bundle_for_path(self, path: Path) -> None:
        self._bundle_manifests[path] = None

    def load_fields(self, path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
        requested_keys = tuple(dict.fromkeys(str(key) for key in keys))
        payload = self._cache.get(path)
        if payload is None:
            self.misses += 1
            payload = {}
            self._cache[path] = payload
        else:
            self.hits += 1
        self._cache.move_to_end(path)

        missing_keys = [key for key in requested_keys if key not in payload]
        if missing_keys:
            manifest = self._manifest_for(path)
            npz_payload: dict[str, np.ndarray] | None = None
            for key in missing_keys:
                bundle_file = self._bundle_path_for_key(path, key) if manifest is not None else None
                if bundle_file is not None and bundle_file.exists():
                    allow_pickle = bool(manifest["keys"][key].get("allow_pickle", False))
                    mmap_mode = None if allow_pickle else "r"
                    try:
                        payload[key] = np.load(bundle_file, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
                        continue
                    except (OSError, EOFError, ValueError):
                        # Fall back to the source npz when a sidecar bundle is truncated or otherwise unreadable.
                        self._disable_bundle_for_path(path)
                        manifest = None
                if npz_payload is None:
                    with np.load(path, allow_pickle=True) as data:
                        npz_payload = {name: data[name] for name in data.files}
                if key in npz_payload:
                    payload[key] = npz_payload[key]
            self._cache.move_to_end(path)
            while len(self._cache) > self.max_open_shards:
                self._cache.popitem(last=False)
                self.evictions += 1
        return {key: payload[key] for key in requested_keys if key in payload}

    def stats(self) -> dict[str, int]:
        return {
            "max_open_shards": self.max_open_shards,
            "open_shards": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class RowShardAccessor:
    def __init__(self, mode_dir: Path, shard_pattern: str, cache: NpzShardCache, *, keys: Iterable[str]) -> None:
        self.mode_dir = mode_dir
        self.shard_pattern = shard_pattern
        self.cache = cache
        self.keys = tuple(dict.fromkeys(str(key) for key in keys))

    def get_row(self, shard_id: int, row_in_shard: int) -> dict[str, Any]:
        shard_path = self.mode_dir / self.shard_pattern.format(shard_id=shard_id)
        payload = self.cache.load_fields(shard_path, self.keys)
        row: dict[str, Any] = {}
        for key, value in payload.items():
            if getattr(value, "ndim", 0) == 0:
                continue
            if value.shape[0] <= row_in_shard:
                continue
            row[key] = value[row_in_shard]
        return row


class RangeShardAccessor:
    def __init__(self, mode_dir: Path, shard_pattern: str, shard_size: int, cache: NpzShardCache, *, keys: Iterable[str]) -> None:
        self.mode_dir = mode_dir
        self.shard_pattern = shard_pattern
        self.shard_size = int(shard_size)
        self.cache = cache
        self.keys = tuple(dict.fromkeys(str(key) for key in keys))

    def get_range(self, start: int, end: int) -> dict[str, np.ndarray]:
        if end <= start:
            return {}
        merged: dict[str, list[np.ndarray]] = {}
        cursor = start
        while cursor < end:
            shard_id = cursor // self.shard_size
            local_start = cursor % self.shard_size
            local_end = min(end - shard_id * self.shard_size, self.shard_size)
            shard_path = self.mode_dir / self.shard_pattern.format(shard_id=shard_id)
            payload = self.cache.load_fields(shard_path, self.keys)
            for key, value in payload.items():
                merged.setdefault(key, []).append(value[local_start:local_end])
            cursor = shard_id * self.shard_size + local_end
        return {
            key: parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            for key, parts in merged.items()
        }

    def get_slice(self, shard_id: int, local_start: int, local_end: int) -> dict[str, np.ndarray]:
        if local_end <= local_start:
            return {}
        shard_path = self.mode_dir / self.shard_pattern.format(shard_id=shard_id)
        payload = self.cache.load_fields(shard_path, self.keys)
        return {key: value[local_start:local_end] for key, value in payload.items()}
