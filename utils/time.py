from __future__ import annotations

from datetime import datetime, timezone
from math import log


def parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def seconds_between(start: str | None, end: str | None) -> float | None:
    start_dt = parse_iso_utc(start)
    end_dt = parse_iso_utc(end)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds()


def safe_log_time(seconds: float | None) -> float | None:
    if seconds is None or seconds < 0:
        return None
    return log(seconds + 1.0)
