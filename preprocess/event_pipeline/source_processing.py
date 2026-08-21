from __future__ import annotations

from preprocess.event_pipeline.schema import SourceView
from preprocess.event_pipeline.text_processing import preprocess_text_bundle
from utils.time import seconds_between


def preprocess_source_view(view: SourceView, source_event_time: str | None) -> SourceView:
    view.profile.profile_description = preprocess_text_bundle(view.profile.profile_description)
    view.profile.display_name = preprocess_text_bundle(view.profile.display_name)
    if view.profile.account_created_at and source_event_time:
        delta = seconds_between(view.profile.account_created_at, source_event_time)
        if delta is not None:
            view.profile.account_age_days = delta / 86400.0
    return view
