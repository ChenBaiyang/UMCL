from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_content_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "event_ids": [row["event_id"] for row in rows],
        "input_ids": torch.as_tensor(np.stack([row["input_ids"] for row in rows], axis=0), dtype=torch.long),
        "attention_mask": torch.as_tensor(np.stack([row["attention_mask"] for row in rows], axis=0), dtype=torch.long),
    }
    if "surface_features" in rows[0]:
        batch["surface_features"] = torch.as_tensor(np.stack([row["surface_features"] for row in rows], axis=0), dtype=torch.float32)
    if "quality_flags" in rows[0]:
        batch["quality_flags"] = torch.as_tensor(np.stack([row["quality_flags"] for row in rows], axis=0), dtype=torch.float32)
    if "primary_label_ids" in rows[0]:
        batch["primary_label_ids"] = torch.as_tensor(np.asarray([row["primary_label_ids"] for row in rows]), dtype=torch.long)
    return batch
