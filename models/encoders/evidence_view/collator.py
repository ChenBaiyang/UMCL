from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_evidence_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(rows)
    max_refs = max((row["ref_row_features"].shape[0] for row in rows), default=0)
    ref_dim = rows[0]["ref_row_features"].shape[1] if rows and rows[0]["ref_row_features"].ndim == 2 else 0
    padded_refs = np.zeros((batch_size, max_refs, ref_dim), dtype=np.float32)
    padded_mask = np.zeros((batch_size, max_refs), dtype=np.uint8)
    for idx, row in enumerate(rows):
        count = row["ref_row_features"].shape[0]
        if count:
            padded_refs[idx, :count] = row["ref_row_features"]
            padded_mask[idx, :count] = 1
    return {
        "event_ids": [row["event_id"] for row in rows],
        "summary_features": torch.as_tensor(np.stack([row["summary_features"] for row in rows], axis=0), dtype=torch.float32),
        "summary_quality_flags": torch.as_tensor(np.stack([row["summary_quality_flags"] for row in rows], axis=0), dtype=torch.float32),
        "primary_label_ids": torch.as_tensor(np.asarray([row["primary_label_ids"] for row in rows]), dtype=torch.long),
        "ref_row_features": torch.as_tensor(padded_refs, dtype=torch.float32),
        "ref_row_mask": torch.as_tensor(padded_mask, dtype=torch.float32),
    }
