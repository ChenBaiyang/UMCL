from __future__ import annotations

from collections import defaultdict
from typing import Any

from preprocess.event_pipeline.availability import recompute_view_availability
from preprocess.event_pipeline.schema import (
    ContentSurfaceMeta,
    ContentView,
    EngagementBundle,
    EvidenceReference,
    EvidenceSummary,
    EvidenceView,
    PropagationEdge,
    PropagationNode,
    PropagationView,
    ReplyItem,
    SourceMissingness,
    SourceProfile,
    SourceView,
    StanceView,
    TaskTarget,
    TextBundle,
    TimeBundle,
    UnifiedEvent,
)
from utils.ids import stable_sort_key


ALLOWED_STANCE_LABELS = {"support", "deny", "query", "comment"}
ALLOWED_INTERACTIONS = {"source", "reply", "repost", "quote_repost"}
ALLOWED_TEXT_STATUS = {"substantive", "boilerplate", "empty", "placeholder", "unknown"}


def _map_interaction_type(value: str | None) -> str:
    return value if value in ALLOWED_INTERACTIONS else "unknown"


def _is_placeholder(dataset_id: str, post: dict[str, Any]) -> bool:
    if post.get("text_status") == "placeholder":
        return True
    extra = post.get("extra", {})
    return dataset_id == "RumourEval" and (extra.get("kind") == "more" or extra.get("reddit_kind") == "more")


def _map_text_status(dataset_id: str, post: dict[str, Any]) -> str:
    if _is_placeholder(dataset_id, post):
        return "placeholder"
    value = post.get("text_status")
    return value if value in ALLOWED_TEXT_STATUS else "unknown"


def _engagement_bundle(payload: dict[str, Any]) -> EngagementBundle:
    return EngagementBundle(
        like_count=payload.get("like_count"),
        reply_count=payload.get("reply_count"),
        repost_count=payload.get("repost_count"),
    )


def _time_bundle(post: dict[str, Any]) -> TimeBundle:
    if post.get("created_at"):
        return TimeBundle(
            observation_kind="absolute_timestamp",
            created_at=post["created_at"],
            is_observed_time=True,
        )
    extra = post.get("extra", {})
    raw_values = extra.get("time_delay_minutes_raw_values")
    bucketed_min = extra.get("time_delay_minutes_bucketed")
    bucketed_hr = extra.get("time_delay_hours_bucketed")
    if raw_values is not None or bucketed_min is not None or bucketed_hr is not None:
        min_list = [str(bucketed_min)] if bucketed_min is not None and not isinstance(bucketed_min, list) else bucketed_min
        hr_list = [str(bucketed_hr)] if bucketed_hr is not None and not isinstance(bucketed_hr, list) else bucketed_hr
        numeric_raw = None
        if raw_values is not None:
            numeric_raw = []
            for item in raw_values:
                try:
                    numeric_raw.append(float(item))
                except (TypeError, ValueError):
                    continue
        return TimeBundle(
            observation_kind="relative_delay_observed",
            time_delay_minutes_raw_values=numeric_raw,
            time_delay_minutes_bucketed=min_list,
            time_delay_hours_bucketed=hr_list,
            is_observed_time=True,
        )
    return TimeBundle(observation_kind="missing", is_observed_time=False)


def _text_bundle(dataset_id: str, post: dict[str, Any]) -> TextBundle:
    return TextBundle(
        raw=post.get("text_raw"),
        lang=post.get("lang"),
        status=_map_text_status(dataset_id, post),
    )


def _reply_extra(post: dict[str, Any]) -> dict[str, Any]:
    extra = dict(post.get("extra", {}))
    if "kind" in extra:
        extra["reddit_kind"] = extra.pop("kind")
    return extra


class UnifiedEventBuilder:
    """Build one model-agnostic UnifiedEvent from one canonical event."""

    def __init__(self, dataset, annotations: list[dict[str, Any]] | None = None) -> None:
        self.dataset = dataset
        self._event_annotations_by_target = defaultdict(list)
        self._post_annotations_by_target = defaultdict(list)
        source_annotations = annotations if annotations is not None else self.dataset.iter_annotations()
        for ann in source_annotations:
            if ann["target_type"] == "event":
                self._event_annotations_by_target[ann["target_id"]].append(ann)
            elif ann["target_type"] == "post":
                self._post_annotations_by_target[ann["target_id"]].append(ann)

    def build_all(self) -> list[UnifiedEvent]:
        return [self.build_from_event_row(event_row) for event_row in self.dataset.iter_events()]

    def build_from_event_row(self, event_row: dict[str, Any]) -> UnifiedEvent:
        source_post = self.dataset.get("posts", event_row["source_post_id"])
        if source_post is None:
            raise ValueError(f"Missing source post for event {event_row['event_id']}")

        posts = list(self.dataset.iter_posts(event_id=event_row["event_id"]))
        posts_by_id = {post["post_id"]: post for post in posts}
        source_user = self.dataset.get("users", source_post["user_id"]) if source_post.get("user_id") else None
        source_refs = list(self.dataset.iter_references(post_id=source_post["post_id"]))
        all_refs = list(self.dataset.iter_references(event_id=event_row["event_id"]))

        event_annotations = list(self._event_annotations_by_target.get(event_row["event_id"], []))
        post_annotations = {
            post_id: list(self._post_annotations_by_target.get(post_id, []))
            for post_id in posts_by_id
        }

        unified = UnifiedEvent(
            event_id=event_row["event_id"],
            dataset_id=event_row["dataset_id"],
            platform=event_row.get("platform") or "unknown",
            canonical_split=event_row.get("split") or "unknown",
            raw_split_source=event_row.get("extra", {}).get("raw_split_source"),
            provenance={"event_extra": dict(event_row.get("extra", {}))},
            extra={},
        )
        unified.content = self._build_content(source_post)
        unified.stance = self._build_stance(source_post, posts, post_annotations)
        unified.propagation = self._build_propagation(source_post, posts)
        unified.evidence = self._build_evidence(source_post, source_refs, all_refs)
        unified.source = self._build_source(source_user)
        unified.task_targets = self._build_event_targets(event_annotations)
        recompute_view_availability(unified)
        return unified

    def _build_content(self, source_post: dict[str, Any]) -> ContentView:
        return ContentView(
            source_post_id=source_post["post_id"],
            source_user_id=source_post.get("user_id"),
            interaction_type=_map_interaction_type(source_post.get("interaction_type")),
            text=_text_bundle(self.dataset.dataset_id, source_post),
            time=_time_bundle(source_post),
            surface=ContentSurfaceMeta(
                has_url=source_post.get("has_url"),
                has_media=source_post.get("has_media"),
                engagement=_engagement_bundle(source_post.get("engagement", {})),
                platform_extra=dict(source_post.get("extra", {})),
            ),
        )

    def _build_stance(
        self,
        source_post: dict[str, Any],
        posts: list[dict[str, Any]],
        post_annotations: dict[str, list[dict[str, Any]]],
    ) -> StanceView:
        reply_items: list[ReplyItem] = []
        source_id = source_post["post_id"]
        ordered_posts = sorted(
            (post for post in posts if post["post_id"] != source_id),
            key=lambda item: stable_sort_key(item.get("created_at"), item["post_id"]),
        )
        for index, post in enumerate(ordered_posts):
            # 修复：support multiple stance annotation field names
            # Different canonical data versions may use different task names for stance labels
            stance_task_names = {"stance", "reply_stance", "stance_label", "rumor_stance"}
            anns = [ann for ann in post_annotations.get(post["post_id"], []) if ann.get("task") in stance_task_names]
            stance_label = anns[0]["label"] if anns and anns[0]["label"] in ALLOWED_STANCE_LABELS else None
            stance_label_source = anns[0]["annotator_kind"] if anns else "none"
            text = _text_bundle(self.dataset.dataset_id, post)
            is_placeholder = text.status == "placeholder"
            is_observed_text = text.raw is not None and text.status not in {"empty", "placeholder", "unknown"}
            is_boilerplate = text.status == "boilerplate"
            is_substantive = text.status == "substantive"
            reply_items.append(
                ReplyItem(
                    post_id=post["post_id"],
                    parent_post_id=post.get("parent_post_id"),
                    user_id=post.get("user_id"),
                    interaction_type=_map_interaction_type(post.get("interaction_type")),
                    text=text,
                    time=_time_bundle(post),
                    stance_label=stance_label,
                    stance_label_source=stance_label_source if stance_label_source in {"gold", "silver", "proxy", "mapped"} else "none",
                    is_placeholder=is_placeholder,
                    is_observed_text=is_observed_text,
                    is_semantically_substantive=is_substantive,
                    is_boilerplate_text=is_boilerplate,
                    stable_order=index,
                    extra=_reply_extra(post),
                )
            )
        return StanceView(source_post_id=source_id, reply_items=reply_items)

    def _build_propagation(self, source_post: dict[str, Any], posts: list[dict[str, Any]]) -> PropagationView:
        nodes: list[PropagationNode] = []
        edges: list[PropagationEdge] = []
        source_id = source_post["post_id"]
        ordered_posts = sorted(posts, key=lambda item: stable_sort_key(item.get("created_at"), item["post_id"]))
        for index, post in enumerate(ordered_posts):
            interaction_type = _map_interaction_type(post.get("interaction_type"))
            text_status = _map_text_status(self.dataset.dataset_id, post)
            is_placeholder = text_status == "placeholder"
            is_observed_text = post.get("text_raw") is not None and text_status not in {"empty", "placeholder", "unknown"}
            parent_inferred = bool(post.get("extra", {}).get("parent_inferred"))
            nodes.append(
                PropagationNode(
                    post_id=post["post_id"],
                    parent_post_id=post.get("parent_post_id"),
                    is_source=(post["post_id"] == source_id),
                    interaction_type=interaction_type,
                    text_status=text_status,
                    is_placeholder=is_placeholder,
                    is_observed_text=is_observed_text,
                    time=_time_bundle(post),
                    stable_order=index,
                    parent_inferred=parent_inferred,
                    extra=dict(post.get("extra", {})),
                )
            )
            if post.get("parent_post_id"):
                edges.append(
                    PropagationEdge(
                        parent_post_id=post["parent_post_id"],
                        child_post_id=post["post_id"],
                    )
                )
        return PropagationView(
            root_post_id=source_post["root_post_id"],
            source_post_id=source_id,
            source_time=_time_bundle(source_post),
            nodes=nodes,
            edges=edges,
        )

    def _build_evidence(
        self,
        source_post: dict[str, Any],
        source_refs: list[dict[str, Any]],
        all_refs: list[dict[str, Any]],
    ) -> EvidenceView:
        references = [
            EvidenceReference(
                ref_id=ref["ref_id"],
                ref_type=ref["ref_type"],
                raw_value=ref.get("raw_value"),
                normalized_value=ref.get("normalized_value"),
                domain=ref.get("domain"),
                target_post_id=ref.get("target_post_id"),
                extra=dict(ref.get("extra", {})),
            )
            for ref in sorted(source_refs, key=lambda item: item["ref_id"])
        ]
        summary = EvidenceSummary(
            has_explicit_ref=bool(references),
            n_refs_total=len(references),
            n_url_refs=sum(1 for ref in references if ref.ref_type == "url"),
            n_media_refs=sum(1 for ref in references if ref.ref_type == "media"),
            n_internal_refs=sum(1 for ref in references if ref.ref_type in {"quoted_post", "reposted_post"}),
            n_external_refs=sum(1 for ref in references if ref.ref_type in {"url", "media"}),
            thread_n_refs_total=len(all_refs),
        )
        return EvidenceView(source_post_id=source_post["post_id"], references=references, summary=summary)

    def _build_source(self, source_user: dict[str, Any] | None) -> SourceView:
        if source_user is None:
            return SourceView(source_user_id=None)
        missingness = SourceMissingness(
            account_created_at="observed" if source_user.get("account_created_at") is not None else "missing",
            verified="observed" if source_user.get("verified") is not None else "missing",
            followers_count="observed" if source_user.get("followers_count") is not None else "missing",
            following_count="observed" if source_user.get("following_count") is not None else "missing",
            post_count="observed" if source_user.get("post_count") is not None else "missing",
            profile_description_raw="observed" if source_user.get("profile_description_raw") is not None else "missing",
            display_name_raw="observed" if source_user.get("display_name_raw") is not None else "missing",
        )
        return SourceView(
            source_user_id=source_user["user_id"],
            profile=SourceProfile(
                account_created_at=source_user.get("account_created_at"),
                verified=source_user.get("verified"),
                followers_count=source_user.get("followers_count"),
                following_count=source_user.get("following_count"),
                post_count=source_user.get("post_count"),
                profile_description=TextBundle(raw=source_user.get("profile_description_raw")),
                display_name=TextBundle(raw=source_user.get("display_name_raw")),
                missingness=missingness,
                platform_extra=dict(source_user.get("extra", {})),
            ),
        )

    def _build_event_targets(self, annotations: list[dict[str, Any]]) -> dict[str, TaskTarget]:
        targets: dict[str, TaskTarget] = {}
        for ann in sorted(annotations, key=lambda item: item["annotation_id"]):
            task = ann["task"] if ann["task"] in {"veracity", "rumour_status", "stance"} else "unknown"
            targets[task] = TaskTarget(
                task=task,
                label=ann.get("label"),
                label_space=ann.get("label_space"),
                original_label=ann.get("original_label"),
                annotator_kind=ann.get("annotator_kind") if ann.get("annotator_kind") in {"gold", "silver", "proxy", "mapped"} else "none",
                confidence=ann.get("confidence"),
            )
        return targets
