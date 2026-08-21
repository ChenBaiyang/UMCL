from __future__ import annotations

from typing import Any, Optional

from utils.primary_label_space import MISSING_LABEL_ID, NON_RUMOUR_CANONICAL, align_primary_label

PRIMARY_LABEL_4WAY_NAMES = ["true", "false", "unverified", NON_RUMOUR_CANONICAL]
PRIMARY_LABEL_2WAY_NAMES = ["rumour", NON_RUMOUR_CANONICAL]
STANCE_LABEL_NAMES = ["support", "deny", "query", "comment"]
STANCE_LABEL_TO_ID = {name: idx for idx, name in enumerate(STANCE_LABEL_NAMES)}


def align_event_labels(dataset_id: str, task_targets: dict[str, Any], data_root: str = "data") -> dict[str, object]:
    return align_primary_label(dataset_id, task_targets, data_root=data_root)


def align_reply_label(label_name: Optional[str]) -> tuple[Optional[str], int, int]:
    if label_name is None:
        return None, MISSING_LABEL_ID, 0
    return label_name, STANCE_LABEL_TO_ID.get(label_name, MISSING_LABEL_ID), 1
