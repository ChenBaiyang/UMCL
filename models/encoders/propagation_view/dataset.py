from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from utils.preprocessed_store import iter_preprocessed_events as iter_preprocessed_event_rows


def iter_preprocessed_events(path: Path):
    yield from iter_preprocessed_event_rows(path)


class PropagationShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with np.load(shard_path, allow_pickle=True) as data:
            self.payload = {key: data[key] for key in data.files}
        self.length = int(self.payload["node_features_continuous"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "event_id": self.payload["event_ids"][index].item() if hasattr(self.payload["event_ids"][index], "item") else self.payload["event_ids"][index],
            "node_post_id": self.payload["node_post_ids"][index].item()
            if hasattr(self.payload["node_post_ids"][index], "item")
            else self.payload["node_post_ids"][index],
            "node_features_continuous": self.payload["node_features_continuous"][index],
            "node_features_categorical": self.payload["node_features_categorical"][index],
            "node_masks": self.payload["node_masks"][index],
            "parent_index": self.payload["parent_index"][index],
            "event_index": self.payload["event_index"][index],
            "label_ids": self.payload["label_ids"][index],
            "label_mask": self.payload["label_mask"][index],
        }
