from __future__ import annotations

from statistics import mean

from preprocess.event_pipeline.schema import TimeBundle
from utils.time import safe_log_time, seconds_between


def _first_float(values: list[str] | list[float] | None) -> float | None:
    if not values:
        return None
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return mean(numeric)


def preprocess_time_bundle(time: TimeBundle, source_created_at: str | None, enable_log: bool) -> TimeBundle:
    if time.observation_kind == "absolute_timestamp":
        delta = seconds_between(source_created_at, time.created_at)
        time.time_from_source_sec = delta
        time.time_from_source_min = None if delta is None else delta / 60.0
        time.is_observed_time = True
    elif time.observation_kind == "relative_delay_observed":
        raw_minutes = _first_float(time.time_delay_minutes_raw_values)
        if raw_minutes is None:
            raw_minutes = _first_float(time.time_delay_minutes_bucketed)
        if raw_minutes is None:
            raw_minutes = _first_float(time.time_delay_hours_bucketed)
            if raw_minutes is not None:
                raw_minutes *= 60.0
        time.time_from_source_min = raw_minutes
        time.time_from_source_sec = None if raw_minutes is None else raw_minutes * 60.0
        time.is_observed_time = True
    else:
        time.is_observed_time = False

    if enable_log:
        time.log_time_from_source_sec = safe_log_time(time.time_from_source_sec)
    return time
