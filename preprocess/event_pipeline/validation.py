from __future__ import annotations

from collections import Counter

from preprocess.event_pipeline.schema import UnifiedEvent


def validate_unified_event(event: UnifiedEvent) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if not event.content:
        errors.append("missing content view")
        return warnings, errors

    source_post_id = event.content.source_post_id
    if not event.propagation:
        errors.append("missing propagation view")
    else:
        source_nodes = [node for node in event.propagation.nodes if node.is_source]
        if len(source_nodes) != 1:
            errors.append(f"expected exactly one source node, got {len(source_nodes)}")
        roots = {node.post_id for node in event.propagation.nodes if node.is_source}
        if roots and source_post_id not in roots:
            errors.append("source post id does not match propagation source node")
        for node in event.propagation.nodes:
            if node.post_id != source_post_id and node.parent_post_id is None:
                warnings.append(f"node without parent: {node.post_id}")

    if event.stance:
        for reply in event.stance.reply_items:
            if reply.is_placeholder and reply.text.status != "placeholder":
                errors.append(f"placeholder reply without placeholder text_status: {reply.post_id}")
            if not reply.is_observed_text and reply.text.status == "substantive":
                warnings.append(f"substantive text but is_observed_text=false: {reply.post_id}")

    if event.source:
        profile = event.source.profile
        missingness = profile.missingness
        pairs = {
            "account_created_at": profile.account_created_at,
            "verified": profile.verified,
            "followers_count": profile.followers_count,
            "following_count": profile.following_count,
            "post_count": profile.post_count,
            "profile_description_raw": profile.profile_description.raw,
            "display_name_raw": profile.display_name.raw,
        }
        for key, value in pairs.items():
            status = getattr(missingness, key)
            if value is None and status != "missing":
                errors.append(f"source missingness mismatch for {key}")
            if value is not None and status != "observed":
                errors.append(f"source missingness mismatch for {key}")

    if event.dataset_id == "Weibo":
        interaction_counts = Counter(
            reply.interaction_type for reply in (event.stance.reply_items if event.stance else [])
        )
        if "quote_repost" not in interaction_counts and "repost" not in interaction_counts:
            warnings.append("weibo event without repost-like responses")

    if event.dataset_id in {"Twitter15", "Twitter16"}:
        has_relative_delay = False
        for node in event.propagation.nodes if event.propagation else []:
            if node.time.observation_kind == "relative_delay_observed":
                has_relative_delay = True
                break
        if event.propagation and event.propagation.nodes and not has_relative_delay:
            warnings.append("twitter15/16 event without relative delay observations")

    if event.observed_view_mask.get("evidence", 0) == 1 and not event.evidence:
        errors.append("evidence mask observed but evidence view missing")

    return warnings, errors
