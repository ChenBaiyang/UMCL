from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_source_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_ids": [row["event_id"] for row in rows],
        "summary_features": torch.as_tensor(np.stack([row["summary_features"] for row in rows], axis=0), dtype=torch.float32),
        "summary_quality_flags": torch.as_tensor(np.stack([row["summary_quality_flags"] for row in rows], axis=0), dtype=torch.float32),
        "categorical_features": torch.as_tensor(np.stack([row["categorical_features"] for row in rows], axis=0), dtype=torch.long),
        "primary_label_ids": torch.as_tensor(np.asarray([row["primary_label_ids"] for row in rows]), dtype=torch.long),
    }
