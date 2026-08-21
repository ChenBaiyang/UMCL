from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config.types import Mode


@dataclass
class EventPipelineConfig:
    dataset_id: str | None = None
    mode: Mode = "full"
    optional_log_time_from_source: bool = False
    split_seed: int = 13
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
