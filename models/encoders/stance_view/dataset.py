from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - allows non-training utilities to run without torch installed
    class Dataset:  # type: ignore[no-redef]
        pass

from utils.preprocessed_store import iter_preprocessed_events as iter_preprocessed_event_rows


def iter_preprocessed_events(path: Path):
    yield from iter_preprocessed_event_rows(path)


class StanceShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with np.load(shard_path, allow_pickle=False) as data:
            self.payload = {key: data[key] for key in data.files}
        self.length = int(self.payload["reply_embedding"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {
            "reply_embedding": self.payload["reply_embedding"][index],
            "context_cat": self.payload["context_cat"][index],
            "context_dense": self.payload["context_dense"][index],
            "reply_label_ids": self.payload["reply_label_ids"][index],
            "reply_label_mask": self.payload["reply_label_mask"][index],
            "event_index": self.payload["event_index"][index],
            "event_sampled_reply_count": self.payload["event_sampled_reply_count"][index],
            "event_truncation_flag": self.payload["event_truncation_flag"][index],
        }
        if "reply_id" in self.payload:
            row["reply_id"] = (
                self.payload["reply_id"][index].item()
                if hasattr(self.payload["reply_id"][index], "item")
                else self.payload["reply_id"][index]
            )
        if "parent_post_id" in self.payload:
            row["parent_post_id"] = (
                self.payload["parent_post_id"][index].item()
                if hasattr(self.payload["parent_post_id"][index], "item")
                else self.payload["parent_post_id"][index]
            )
        if "source_post_id" in self.payload:
            row["source_post_id"] = (
                self.payload["source_post_id"][index].item()
                if hasattr(self.payload["source_post_id"][index], "item")
                else self.payload["source_post_id"][index]
            )
        if "dataset_row_index" in self.payload:
            row["dataset_row_index"] = self.payload["dataset_row_index"][index]
        return row
