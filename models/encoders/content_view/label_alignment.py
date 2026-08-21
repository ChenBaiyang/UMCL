from __future__ import annotations

from typing import Any

from utils.primary_label_space import MISSING_LABEL_ID, align_primary_label


def align_content_labels(
    dataset_id: str,
    task_targets: dict[str, Any],
    data_root: str = "data",
) -> dict[str, object]:
    return align_primary_label(dataset_id, task_targets, data_root=data_root)


__all__ = ["MISSING_LABEL_ID", "align_content_labels"]
