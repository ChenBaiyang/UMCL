from __future__ import annotations

from preprocess.event_pipeline.schema import UnifiedEvent, ViewAvailability


def recompute_view_availability(event: UnifiedEvent) -> None:
    source_has_text = bool(event.content and event.content.text.raw is not None)
    has_non_source = bool(event.stance and event.stance.reply_items)
    has_nodes = bool(event.propagation and event.propagation.nodes)
    has_refs = bool(event.evidence and event.evidence.references)
    source_surface_ref = bool(
        event.content
        and (
            event.content.surface.has_url is True
            or event.content.surface.has_media is True
        )
    )
    has_source_user = bool(event.source and event.source.source_user_id)

    event.observed_view_mask = {
        "content": int(source_has_text),
        "stance": int(has_non_source),
        "propagation": int(has_nodes),
        "evidence": int(has_refs or source_surface_ref),
        "source": int(has_source_user),
    }
    event.view_availability = {
        "content": ViewAvailability(
            observed=source_has_text,
            source_type="source_post_text" if source_has_text else "missing",
            notes=[],
        ),
        "stance": ViewAvailability(
            observed=has_non_source,
            source_type="reply_posts" if has_non_source else "missing",
            notes=[],
        ),
        "propagation": ViewAvailability(
            observed=has_nodes,
            source_type="event_posts" if has_nodes else "missing",
            notes=[],
        ),
        "evidence": ViewAvailability(
            observed=(has_refs or source_surface_ref),
            source_type="source_refs" if has_refs else ("source_surface" if source_surface_ref else "missing"),
            notes=[],
        ),
        "source": ViewAvailability(
            observed=has_source_user,
            source_type="source_user" if has_source_user else "missing",
            notes=[],
        ),
    }
