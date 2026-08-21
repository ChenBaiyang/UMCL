from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class _IdRegistry:
    next_id: int = 1
    string_to_int: dict[str, int] = field(default_factory=dict)
    int_to_string: dict[int, str] = field(default_factory=dict)

    def get_or_add(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        key = str(value)
        if key not in self.string_to_int:
            assigned = self.next_id
            self.next_id += 1
            self.string_to_int[key] = assigned
            self.int_to_string[assigned] = key
        return self.string_to_int[key]

    def export(self) -> dict[str, str]:
        return {str(int_id): raw_id for int_id, raw_id in sorted(self.int_to_string.items())}

    def export_compact(self) -> list[str]:
        if not self.int_to_string:
            return []
        return [self.int_to_string[int_id] for int_id in range(1, self.next_id)]


class IdCompactor:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.events = _IdRegistry()
        self.posts = _IdRegistry()
        self.users = _IdRegistry()
        self.refs = _IdRegistry()
        self.annotations = _IdRegistry()

    def prime_annotation_ids(self, annotations: Iterable[dict[str, Any]]) -> None:
        for ann in annotations:
            self.annotations.get_or_add(ann.get("annotation_id"))

    def compact_event_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row["event_id"] = self.events.get_or_add(row.get("event_id"))

        content = row.get("content") or {}
        if content:
            content["source_post_id"] = self.posts.get_or_add(content.get("source_post_id"))
            content["source_user_id"] = self.users.get_or_add(content.get("source_user_id"))

        stance = row.get("stance") or {}
        if stance:
            stance["source_post_id"] = self.posts.get_or_add(stance.get("source_post_id"))
            for reply in stance.get("reply_items") or []:
                reply["post_id"] = self.posts.get_or_add(reply.get("post_id"))
                reply["parent_post_id"] = self.posts.get_or_add(reply.get("parent_post_id"))
                reply["user_id"] = self.users.get_or_add(reply.get("user_id"))

        propagation = row.get("propagation") or {}
        if propagation:
            propagation["root_post_id"] = self.posts.get_or_add(propagation.get("root_post_id"))
            propagation["source_post_id"] = self.posts.get_or_add(propagation.get("source_post_id"))
            for node in propagation.get("nodes") or []:
                node["post_id"] = self.posts.get_or_add(node.get("post_id"))
                node["parent_post_id"] = self.posts.get_or_add(node.get("parent_post_id"))
            for edge in propagation.get("edges") or []:
                edge["parent_post_id"] = self.posts.get_or_add(edge.get("parent_post_id"))
                edge["child_post_id"] = self.posts.get_or_add(edge.get("child_post_id"))

        evidence = row.get("evidence") or {}
        if evidence:
            evidence["source_post_id"] = self.posts.get_or_add(evidence.get("source_post_id"))
            for ref in evidence.get("references") or []:
                ref["ref_id"] = self.refs.get_or_add(ref.get("ref_id"))
                ref["target_post_id"] = self.posts.get_or_add(ref.get("target_post_id"))

        source = row.get("source") or {}
        if source:
            source["source_user_id"] = self.users.get_or_add(source.get("source_user_id"))

        return row

    def export_sidecar(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "id_spaces": {
                "event_id": self.events.export(),
                "post_id": self.posts.export(),
                "user_id": self.users.export(),
                "ref_id": self.refs.export(),
                "annotation_id": self.annotations.export(),
            },
            "counts": {
                "event_id": len(self.events.int_to_string),
                "post_id": len(self.posts.int_to_string),
                "user_id": len(self.users.int_to_string),
                "ref_id": len(self.refs.int_to_string),
                "annotation_id": len(self.annotations.int_to_string),
            },
        }

    def export_summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "event_id": len(self.events.int_to_string),
                "post_id": len(self.posts.int_to_string),
                "user_id": len(self.users.int_to_string),
                "ref_id": len(self.refs.int_to_string),
                "annotation_id": len(self.annotations.int_to_string),
            },
            "samples": {
                "event_id": list(self.events.export().items())[:10],
                "post_id": list(self.posts.export().items())[:10],
                "user_id": list(self.users.export().items())[:10],
                "ref_id": list(self.refs.export().items())[:10],
                "annotation_id": list(self.annotations.export().items())[:10],
            },
        }

    def export_compact_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "format": "compact_id_map_v1",
            "counts": {
                "event_id": len(self.events.int_to_string),
                "post_id": len(self.posts.int_to_string),
                "user_id": len(self.users.int_to_string),
                "ref_id": len(self.refs.int_to_string),
                "annotation_id": len(self.annotations.int_to_string),
            },
            "id_spaces": {
                "event_id": self.events.export_compact(),
                "post_id": self.posts.export_compact(),
                "user_id": self.users.export_compact(),
                "ref_id": self.refs.export_compact(),
                "annotation_id": self.annotations.export_compact(),
            },
        }
