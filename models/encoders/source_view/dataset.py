from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from utils.preprocessed_store import iter_preprocessed_events as iter_preprocessed_event_rows


def iter_preprocessed_events(path: Path):
    yield from iter_preprocessed_event_rows(path)


class SourceShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with np.load(shard_path, allow_pickle=True) as data:
            self.payload = {key: data[key] for key in data.files}
        self.length = int(self.payload["summary_features"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        event_id = self.payload["event_ids"][index]
        if hasattr(event_id, "item"):
            event_id = event_id.item()
        return {
            "event_id": event_id,
            "summary_features": self.payload["summary_features"][index],
            "summary_quality_flags": self.payload["summary_quality_flags"][index],
            "categorical_features": self.payload["categorical_features"][index],
            "primary_label_ids": self.payload["primary_label_ids"][index],
        }
