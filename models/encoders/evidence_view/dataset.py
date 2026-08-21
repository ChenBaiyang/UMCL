from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from utils.preprocessed_store import iter_preprocessed_events as iter_preprocessed_event_rows


def iter_preprocessed_events(path: Path):
    yield from iter_preprocessed_event_rows(path)


class EvidenceShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with np.load(shard_path, allow_pickle=True) as data:
            self.payload = {key: data[key] for key in data.files}
        self.length = int(self.payload["summary_features"].shape[0])
        self.ref_features = self.payload["ref_row_features"]
        self.ref_index = self.payload["ref_row_event_index"].astype(np.int32)
        self.ref_slices = self._build_ref_slices()

    def _build_ref_slices(self) -> list[np.ndarray]:
        buckets: list[list[int]] = [[] for _ in range(self.length)]
        for row_idx, event_idx in enumerate(self.ref_index.tolist()):
            if 0 <= int(event_idx) < self.length:
                buckets[int(event_idx)].append(row_idx)
        return [np.asarray(rows, dtype=np.int32) for rows in buckets]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref_rows = self.ref_slices[index]
        return {
            "event_id": self.payload["event_ids"][index].item() if hasattr(self.payload["event_ids"][index], "item") else self.payload["event_ids"][index],
            "summary_features": self.payload["summary_features"][index],
            "summary_quality_flags": self.payload["summary_quality_flags"][index],
            "primary_label_ids": self.payload["primary_label_ids"][index],
            "ref_row_features": self.ref_features[ref_rows] if ref_rows.size else np.zeros((0, self.ref_features.shape[1]), dtype=np.float32),
            "ref_row_mask": np.ones((ref_rows.size,), dtype=np.uint8) if ref_rows.size else np.zeros((0,), dtype=np.uint8),
        }
