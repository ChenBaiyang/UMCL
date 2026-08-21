from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_propagation_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_ids": [row["event_id"] for row in rows],
        "node_post_ids": [row["node_post_id"] for row in rows],
        "node_features_continuous": torch.as_tensor(np.stack([row["node_features_continuous"] for row in rows], axis=0), dtype=torch.float32),
        "node_features_categorical": torch.as_tensor(np.stack([row["node_features_categorical"] for row in rows], axis=0), dtype=torch.long),
        "node_masks": torch.as_tensor(np.asarray([row["node_masks"] for row in rows]), dtype=torch.float32),
        "parent_index": torch.as_tensor(np.asarray([row["parent_index"] for row in rows]), dtype=torch.long),
        "event_index": torch.as_tensor(np.asarray([row["event_index"] for row in rows]), dtype=torch.long),
        "label_ids": torch.as_tensor(np.asarray([row["label_ids"] for row in rows]), dtype=torch.long),
        "label_mask": torch.as_tensor(np.asarray([row["label_mask"] for row in rows]), dtype=torch.float32),
    }
