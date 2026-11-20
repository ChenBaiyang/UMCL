from __future__ import annotations

from typing import Literal

Mode = Literal["full"]
DatasetId = Literal['PHEME', 'Twitter15', 'Twitter16', 'Weibo']
StageName = Literal["event_pipeline", "encoders", "fusion"]
ViewName = Literal["content", "stance", "propagation", "evidence", "source"]

DATASET_IDS: list[str] = ['PHEME', 'Twitter15', 'Twitter16', 'Weibo']
VIEW_NAMES: list[str] = ["content", "stance", "propagation", "evidence", "source"]
