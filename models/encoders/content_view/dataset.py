from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from models.encoders.content_view.artifacts import EVENT_INDEX_FILENAME
from utils.preprocessed_store import iter_preprocessed_events as iter_preprocessed_event_rows


def iter_preprocessed_events(path: Path):
    yield from iter_preprocessed_event_rows(path)


def load_event_index(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


class ContentShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with np.load(shard_path, allow_pickle=True) as data:
            self.payload = {key: data[key] for key in data.files}
        self.length = int(self.payload["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "event_id": self.payload["event_ids"][index].item()
            if hasattr(self.payload["event_ids"][index], "item")
            else self.payload["event_ids"][index],
            "input_ids": self.payload["input_ids"][index],
            "attention_mask": self.payload["attention_mask"][index],
        }
        if "surface_features" in self.payload:
            item["surface_features"] = self.payload["surface_features"][index]
        if "quality_flags" in self.payload:
            item["quality_flags"] = self.payload["quality_flags"][index]
        if "primary_label_ids" in self.payload:
            item["primary_label_ids"] = self.payload["primary_label_ids"][index]
        return item
