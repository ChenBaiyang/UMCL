from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_stance2_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reply_embedding": torch.as_tensor(np.stack([row["reply_embedding"] for row in rows], axis=0), dtype=torch.float32),
        "context_cat": torch.as_tensor(np.stack([row["context_cat"] for row in rows], axis=0), dtype=torch.long),
        "context_dense": torch.as_tensor(np.stack([row["context_dense"] for row in rows], axis=0), dtype=torch.float32),
        "reply_label_ids": torch.as_tensor(np.asarray([row["reply_label_ids"] for row in rows]), dtype=torch.long),
        "reply_label_mask": torch.as_tensor(np.asarray([row["reply_label_mask"] for row in rows]), dtype=torch.float32),
        "event_index": torch.as_tensor(np.asarray([row["event_index"] for row in rows]), dtype=torch.long),
        "event_sampled_reply_count": torch.as_tensor(np.asarray([row["event_sampled_reply_count"] for row in rows]), dtype=torch.float32),
        "event_truncation_flag": torch.as_tensor(np.asarray([row["event_truncation_flag"] for row in rows]), dtype=torch.float32),
    }
