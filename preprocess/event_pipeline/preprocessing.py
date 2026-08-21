from __future__ import annotations

from copy import deepcopy

from config.event_pipeline import EventPipelineConfig
from preprocess.event_pipeline.evidence_processing import preprocess_evidence_view
from preprocess.event_pipeline.schema import UnifiedEvent
from preprocess.event_pipeline.source_processing import preprocess_source_view
from preprocess.event_pipeline.structure_processing import preprocess_propagation_view, preprocess_stance_view
from preprocess.event_pipeline.text_processing import preprocess_text_bundle
from preprocess.event_pipeline.time_processing import preprocess_time_bundle
from preprocess.event_pipeline.twitter_delay_processing import preserve_relative_delay_reply
from preprocess.event_pipeline.weibo_processing import mark_weibo_reply_semantics


def preprocess_unified_event(event: UnifiedEvent, config: EventPipelineConfig | None = None) -> UnifiedEvent:
    cfg = config or EventPipelineConfig(dataset_id=event.dataset_id, mode="full")
    text_cache: dict[tuple[object, ...], tuple[object, ...]] = {}
    time_cache: dict[tuple[object, ...], tuple[object, ...]] = {}

    def _preprocess_text_cached(text):
        key = (text.raw, text.status)
        cached = text_cache.get(key)
        if cached is None:
            processed = preprocess_text_bundle(text)
            cached = (
                processed.norm_light,
                processed.char_len,
                processed.token_len_approx,
                deepcopy(processed.quality),
            )
            text_cache[key] = cached
            return processed
        text.norm_light = cached[0]
        text.char_len = cached[1]
        text.token_len_approx = cached[2]
        text.quality = deepcopy(cached[3])
        return text

    def _preprocess_time_cached(time, source_created_at):
        key = (
            time.observation_kind,
            time.created_at,
            tuple(time.time_delay_minutes_raw_values or ()),
            tuple(time.time_delay_minutes_bucketed or ()),
            tuple(time.time_delay_hours_bucketed or ()),
            source_created_at,
            cfg.optional_log_time_from_source,
        )
        cached = time_cache.get(key)
        if cached is None:
            processed = preprocess_time_bundle(
                time,
                source_created_at,
                enable_log=cfg.optional_log_time_from_source,
            )
            cached = (
                processed.time_from_source_sec,
                processed.time_from_source_min,
                processed.log_time_from_source_sec,
                processed.is_observed_time,
            )
            time_cache[key] = cached
            return processed
        time.time_from_source_sec = cached[0]
        time.time_from_source_min = cached[1]
        time.log_time_from_source_sec = cached[2]
        time.is_observed_time = cached[3]
        return time

    if event.content:
        event.content.text = _preprocess_text_cached(event.content.text)
        event.content.time = _preprocess_time_cached(event.content.time, event.content.time.created_at)

    source_created_at = event.content.time.created_at if event.content else None

    if event.source:
        event.source = preprocess_source_view(event.source, source_created_at)

    if event.propagation:
        for node in event.propagation.nodes:
            node.time = _preprocess_time_cached(node.time, source_created_at)
        event.propagation = preprocess_propagation_view(event.propagation)

    if event.stance:
        for reply in event.stance.reply_items:
            reply.text = _preprocess_text_cached(reply.text)
            reply.time = _preprocess_time_cached(reply.time, source_created_at)
            if event.dataset_id == "Weibo":
                mark_weibo_reply_semantics(reply)
            if event.dataset_id in {"Twitter15", "Twitter16"}:
                preserve_relative_delay_reply(reply)
            if reply.text.quality.is_semantically_usable and reply.interaction_type != "repost":
                reply.is_semantically_substantive = True
        if event.propagation:
            event.stance = preprocess_stance_view(event.stance, event.propagation)

    if event.evidence:
        event.evidence = preprocess_evidence_view(event.evidence, event.platform)

    return event
