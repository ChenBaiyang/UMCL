"""Canonical conversion for RumourEval."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from preprocess.canonical.schema import (
    AnnotationRecord,
    DatasetMeta,
    EventRecord,
    PostRecord,
    ReferenceRecord,
    UserRecord,
)
from utils.logging_utils import get_logger


STANCE_LABELS = {"support", "deny", "query", "comment"}
VERACITY_LABELS = {"true", "false", "unverified"}
PHEME_VERACITY_LABELS = {"true", "false", "unverified", "non-rumour"}
PHEME_RUMOUR_STATUS_LABELS = {"rumour", "non-rumour", "unclear"}
URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")
ACL2017_LABELS = {"true", "false", "unverified", "non-rumor"}
TREE_NODE_PATTERN = re.compile(r"\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]")
WEIBO_BOILERPLATE_TEXTS = {
    "转发微博",
    "轉發微博",
    "杞彂寰崥",
    "repost",
    "????",
    "??????",
}
WEIBO_BOILERPLATE_NORMALIZED = {re.sub(r"\s+", "", item.strip()).lower().strip("!！?？。.,，:：;；~～[]【】()（）\"'“”‘’") for item in WEIBO_BOILERPLATE_TEXTS}
CANONICAL_LOGGER = get_logger("canonical_convert")
WEIBO_PROGRESS_EVERY = 25
_ZIPFILE_CACHE: dict[str, zipfile.ZipFile] = {}
_ZIPFILE_NAME_CACHE: dict[str, set[str]] = {}


def normalize_text_marker(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.strip()).lower()
    normalized = normalized.strip("!！?？。.,，:：;；~～[]【】()（）\"'“”‘’")
    return normalized


def classify_text_status(
    text_raw: str | None,
    *,
    missing_policy: str = "unknown",
    boilerplate_texts: set[str] | None = None,
) -> str:
    if text_raw is None:
        return missing_policy
    stripped = text_raw.strip()
    if not stripped:
        return "empty"
    normalized = normalize_text_marker(stripped)
    if boilerplate_texts and normalized in boilerplate_texts:
        return "boilerplate"
    return "substantive"


def validate_post_schema(post: PostRecord, errors: list[str]) -> None:
    if post.post_type == "source":
        if post.interaction_type != "source":
            errors.append(f"Source post {post.post_id} must have interaction_type=source")
        if post.parent_post_id is not None:
            errors.append(f"Source post {post.post_id} has non-null parent_post_id")
        if post.root_post_id != post.post_id:
            errors.append(f"Source post {post.post_id} root_post_id must equal post_id")
        return
    if post.post_type != "response":
        errors.append(f"Post {post.post_id} has invalid post_type {post.post_type}")
        return
    if post.interaction_type not in {"reply", "repost", "quote_repost", "other"}:
        errors.append(f"Response post {post.post_id} has invalid interaction_type {post.interaction_type}")


def refresh_stance_view_status(
    events: list[EventRecord],
    posts_by_id: dict[str, PostRecord],
    annotations: list[AnnotationRecord],
) -> None:
    stance_gold_events: set[str] = set()
    stance_text_events: set[str] = set()
    for annotation in annotations:
        if annotation.task != "stance" or annotation.target_type != "post":
            continue
        post = posts_by_id.get(annotation.target_id)
        if post is not None:
            stance_gold_events.add(post.event_id)
    for post in posts_by_id.values():
        if post.post_type == "response" and post.text_status == "substantive":
            stance_text_events.add(post.event_id)
    for event in events:
        has_gold = event.event_id in stance_gold_events
        has_text = event.event_id in stance_text_events
        if has_gold and has_text:
            event.view_status["stance"] = "gold_and_text"
        elif has_gold:
            event.view_status["stance"] = "gold_only"
        elif has_text:
            event.view_status["stance"] = "text_only"
        else:
            event.view_status["stance"] = "none"


def _zip_member_candidates(inner_path: str) -> list[str]:
    normalized = inner_path.replace("\\", "/").lstrip("/")
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        item = value.replace("\\", "/").lstrip("/")
        if item and item not in seen:
            seen.add(item)
            candidates.append(item)

    add(normalized)
    if "/" in normalized:
        add(normalized.split("/", 1)[1])
    add(Path(normalized).name)
    return candidates


def _open_zip_member(archive_path_str: str, inner_path: str):
    archive = _ZIPFILE_CACHE.get(archive_path_str)
    if archive is None:
        archive = zipfile.ZipFile(archive_path_str, mode="r")
        _ZIPFILE_CACHE[archive_path_str] = archive
        _ZIPFILE_NAME_CACHE[archive_path_str] = set(archive.namelist())
    names = _ZIPFILE_NAME_CACHE[archive_path_str]

    for candidate in _zip_member_candidates(inner_path):
        if candidate in names:
            return archive.open(candidate, mode="r")

    basename = Path(inner_path).name
    suffix_matches = [name for name in names if name.endswith(f"/{basename}") or name == basename]
    if len(suffix_matches) == 1:
        return archive.open(suffix_matches[0], mode="r")
    raise KeyError(inner_path)


def _read_text_from_path(path: Any, encoding: str) -> str:
    if hasattr(path, "read_text"):
        return path.read_text(encoding=encoding)
    path_str = str(path)
    if "!/" in path_str:
        archive_path_str, inner_path = path_str.split("!/", 1)
        archive_path_str = str(Path(archive_path_str).resolve())
        with _open_zip_member(archive_path_str, inner_path) as handle:
            return handle.read().decode(encoding)
    return Path(path_str).resolve().read_text(encoding=encoding)


def _worker_path_string(path: Any) -> str:
    path_str = str(path)
    if "!/" in path_str:
        archive_path_str, inner_path = path_str.split("!/", 1)
        return f"{Path(archive_path_str).resolve()}!/{inner_path}"
    return str(Path(path_str).resolve())


def _safe_read_json_with_error(path: Path | str) -> tuple[Any, str | None]:
    last_error: str | None = None
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"]:
        try:
            return json.loads(_read_text_from_path(path, encoding)), None
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, KeyError, zipfile.BadZipFile) as exc:
            last_error = f"{type(exc).__name__}[{encoding}]: {exc}"
            continue
    return {}, last_error


def safe_read_json(path: Path) -> Any:
    payload, _ = _safe_read_json_with_error(path)
    return payload


def _weibo_to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _weibo_canonical_event_id(event_raw_id: str) -> str:
    return f"Weibo:event:weibo:{event_raw_id}"


def _weibo_canonical_post_id(event_raw_id: str, post_raw_id: str) -> str:
    if event_raw_id == post_raw_id:
        return f"Weibo:post:weibo:{event_raw_id}"
    return f"Weibo:post:weibo:{event_raw_id}:{post_raw_id}"


def _weibo_canonical_user_id(uid: str) -> str:
    return f"Weibo:user:weibo:{uid}"


def _weibo_upsert_user(row: dict[str, Any], users_by_id: dict[str, UserRecord]) -> str | None:
    uid = str(row.get("uid") or "").strip()
    if not uid:
        return None
    user_id = _weibo_canonical_user_id(uid)
    if user_id not in users_by_id:
        account_created_at, raw_created = _parse_unix_time(row.get("user_created_at"))
        users_by_id[user_id] = UserRecord(
            user_id=user_id,
            platform="weibo",
            account_created_at=account_created_at,
            verified=bool(row.get("verified")) if row.get("verified") is not None else None,
            followers_count=_weibo_to_int(row.get("followers_count")),
            following_count=_weibo_to_int(row.get("friends_count")),
            post_count=_weibo_to_int(row.get("statuses_count")),
            profile_description_raw=row.get("user_description") if isinstance(row.get("user_description"), str) else None,
            display_name_raw=row.get("screen_name") if isinstance(row.get("screen_name"), str) else None,
            extra={
                "uid_raw": uid,
                "username": row.get("username"),
                "gender": row.get("gender"),
                "verified_reason": row.get("verified_reason"),
                "verified_type": row.get("verified_type"),
                "bi_followers_count": _weibo_to_int(row.get("bi_followers_count")),
                "location": row.get("location"),
                "raw_created_at": raw_created if raw_created is not None else row.get("user_created_at"),
            },
        )
    return user_id


def _weibo_extract_refs(
    *,
    event_id: str,
    post_id: str,
    row: dict[str, Any],
    parent_post_id: str | None,
    interaction_type: str,
) -> list[ReferenceRecord]:
    refs: list[ReferenceRecord] = []
    ref_index = 0
    seen: set[tuple[str, str]] = set()

    for field_name in ["text", "original_text"]:
        value = row.get(field_name)
        if isinstance(value, str):
            for match in URL_PATTERN.findall(value):
                normalized = normalize_url(match)
                key = ("url", normalized)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(
                    ReferenceRecord(
                        ref_id=f"Weibo:ref:{post_id}:{ref_index}",
                        event_id=event_id,
                        post_id=post_id,
                        ref_type="url",
                        raw_value=match,
                        normalized_value=normalized,
                        domain=urlparse(normalized).netloc.lower() or None,
                        target_post_id=None,
                        extra={"source": field_name},
                    )
                )
                ref_index += 1

    picture = row.get("picture")
    if isinstance(picture, str) and picture.strip():
        normalized = normalize_url(picture)
        refs.append(
            ReferenceRecord(
                ref_id=f"Weibo:ref:{post_id}:{ref_index}",
                event_id=event_id,
                post_id=post_id,
                ref_type="media",
                raw_value=picture,
                normalized_value=normalized,
                domain=urlparse(normalized).netloc.lower() or None,
                target_post_id=None,
                extra={"source": "picture"},
            )
        )
        ref_index += 1

    if parent_post_id and interaction_type in {"repost", "quote_repost"}:
        refs.append(
            ReferenceRecord(
                ref_id=f"Weibo:ref:{post_id}:repost",
                event_id=event_id,
                post_id=post_id,
                ref_type="reposted_post",
                raw_value=None,
                normalized_value=None,
                domain=None,
                target_post_id=parent_post_id,
                extra={"source": "parent_post_id"},
            )
        )

    return refs


def _convert_single_weibo_event(task: tuple[int, str, str, list[str], str]) -> dict[str, Any]:
    index, event_raw_id, label_raw, indexed_post_ids, json_path_str = task
    json_path = Path(json_path_str)
    rows, read_error = _safe_read_json_with_error(json_path_str)
    if not isinstance(rows, list) or not rows:
        return {
            "index": index,
            "event": None,
            "posts": [],
            "users": [],
            "references": [],
            "annotation": None,
            "validation_warnings": [],
            "skipped_items": [
                {
                    "type": "event",
                    "path": str(json_path),
                    "reason": "missing or unreadable event json",
                    "reader_path": json_path_str,
                    "read_error": read_error,
                }
            ],
        }

    event_id = _weibo_canonical_event_id(event_raw_id)
    posts_by_id_raw: dict[str, dict[str, Any]] = {}
    posts_by_mid_raw: dict[str, dict[str, Any]] = {}
    created_times: list[str] = []
    source_row = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("id") or "").strip()
        raw_mid = str(row.get("mid") or "").strip()
        if raw_id:
            posts_by_id_raw[raw_id] = row
        if raw_mid:
            posts_by_mid_raw[raw_mid] = row
        parent_raw = row.get("parent")
        if source_row is None and parent_raw in (None, "", "None", 0, "0"):
            source_row = row

    if source_row is None:
        source_row = rows[0] if rows and isinstance(rows[0], dict) else None
    if source_row is None:
        return {
            "index": index,
            "event": None,
            "posts": [],
            "users": [],
            "references": [],
            "annotation": None,
            "validation_warnings": [],
            "skipped_items": [
                {
                    "type": "event",
                    "path": str(json_path),
                    "reason": "missing source row",
                    "reader_path": json_path_str,
                }
            ],
        }

    source_raw_id = str(source_row.get("id") or event_raw_id).strip()
    source_post_id = _weibo_canonical_post_id(event_raw_id, source_raw_id)
    posts: list[PostRecord] = []
    users_by_id: dict[str, UserRecord] = {}
    references: list[ReferenceRecord] = []
    skipped_items: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("id") or "").strip()
        if not raw_id:
            skipped_items.append({"type": "post", "path": str(json_path), "reason": "missing post id"})
            continue
        post_id = _weibo_canonical_post_id(event_raw_id, raw_id)
        raw_parent = row.get("parent")
        if raw_parent in (None, "", "None", 0, "0") or raw_id == source_raw_id:
            parent_post_id = None if raw_id == source_raw_id else source_post_id
        else:
            parent_key = str(raw_parent).strip()
            parent_row = posts_by_mid_raw.get(parent_key) or posts_by_id_raw.get(parent_key)
            if parent_row is not None:
                parent_post_id = _weibo_canonical_post_id(event_raw_id, str(parent_row.get("id")).strip())
            else:
                parent_post_id = source_post_id

        created_at, raw_time = _parse_unix_time(row.get("t"))
        if created_at:
            created_times.append(created_at)
        text_raw = None
        if isinstance(row.get("text"), str) and row["text"].strip():
            text_raw = row["text"].strip()
        elif isinstance(row.get("original_text"), str) and row["original_text"].strip():
            text_raw = row["original_text"].strip()

        text_status = classify_text_status(
            text_raw,
            missing_policy="unknown" if raw_id == source_raw_id else "empty",
            boilerplate_texts=WEIBO_BOILERPLATE_NORMALIZED,
        )
        interaction_type = "source"
        if raw_id != source_raw_id:
            interaction_type = "quote_repost" if text_status == "substantive" else "repost"

        user_id = _weibo_upsert_user(row, users_by_id)
        references.extend(
            _weibo_extract_refs(
                event_id=event_id,
                post_id=post_id,
                row=row,
                parent_post_id=parent_post_id if raw_id != source_raw_id else None,
                interaction_type=interaction_type,
            )
        )
        posts.append(
            PostRecord(
                post_id=post_id,
                event_id=event_id,
                user_id=user_id,
                post_type="source" if raw_id == source_raw_id else "response",
                interaction_type=interaction_type,
                text_status=text_status,
                parent_post_id=parent_post_id,
                root_post_id=source_post_id,
                created_at=created_at,
                text_raw=text_raw,
                lang="zh",
                engagement={
                    "like_count": _weibo_to_int(row.get("attitudes_count")),
                    "reply_count": _weibo_to_int(row.get("comments_count")),
                    "repost_count": _weibo_to_int(row.get("reposts_count")),
                },
                has_media=bool(row.get("picture")),
                has_url=bool(text_raw and URL_PATTERN.search(text_raw)),
                extra={
                    "event_id_raw": event_raw_id,
                    "post_id_raw": raw_id,
                    "mid_raw": str(row.get("mid") or "").strip() or None,
                    "parent_raw": raw_parent,
                    "raw_time_unix": raw_time if raw_time is not None else row.get("t"),
                    "geo_raw": row.get("geo"),
                    "province": row.get("province"),
                    "city": row.get("city"),
                    "indexed_post_ids_contains_post": raw_id in indexed_post_ids,
                },
            )
        )

    if len(posts) == 1:
        propagation = "flat"
    elif all(
        post.parent_post_id in (None, source_post_id) or post.post_id == source_post_id
        for post in posts
    ):
        propagation = "star"
    else:
        propagation = "full_tree"

    response_text_available = any(post.post_type == "response" and post.text_status == "substantive" for post in posts)
    event = EventRecord(
        event_id=event_id,
        dataset_id="Weibo",
        platform="weibo",
        source_post_id=source_post_id,
        split="all",
        topic=None,
        lang="zh",
        start_time=min(created_times) if created_times else None,
        end_time=max(created_times) if created_times else None,
        view_status={
            "semantic": "available",
            "stance": "text_only" if response_text_available else "none",
            "propagation": propagation,
            "evidence": "none",
            "source": "partial",
        },
        extra={
            "event_id_raw": event_raw_id,
            "json_path": str(json_path),
            "indexed_post_count": len(indexed_post_ids),
            "json_post_count": len([row for row in rows if isinstance(row, dict)]),
            "label_raw": label_raw,
        },
    )

    label = RumdectTwitterCanonicalConverter._map_label(label_raw)
    annotation = None
    validation_warnings: list[str] = []
    if label is None:
        validation_warnings.append(f"Unknown label {label_raw!r} for Weibo event {event_raw_id}")
    else:
        annotation = AnnotationRecord(
            annotation_id=f"Weibo:ann:rumour_status:{event_id}",
            target_type="event",
            target_id=event_id,
            task="rumour_status",
            label=label,
            label_space="rumour_status",
            original_label=label_raw,
            anchor_post_id=source_post_id,
            annotator_kind="gold",
            confidence=None,
            extra={"source_file": "Weibo.txt", "event_id_raw": event_raw_id},
        )

    return {
        "index": index,
        "event": event,
        "posts": posts,
        "users": list(users_by_id.values()),
        "references": references,
        "annotation": annotation,
        "validation_warnings": validation_warnings,
        "skipped_items": skipped_items,
    }


def is_hidden_artifact(path: Path) -> bool:
    return any(part.startswith("._") or part == ".DS_Store" for part in path.parts)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unwrap_reddit_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        kind = payload.get("kind")
        data = payload.get("data")
        if kind in {"t1", "t3"} and isinstance(data, dict):
            return data
        if kind == "more" and isinstance(data, dict):
            enriched = dict(data)
            enriched["_reddit_kind"] = "more"
            return enriched
        if kind == "Listing" and isinstance(data, dict):
            children = data.get("children") or []
            if children:
                return _unwrap_reddit_payload(children[0])
        return payload
    if isinstance(payload, list) and payload:
        for item in payload:
            candidate = _unwrap_reddit_payload(item)
            if isinstance(candidate, dict) and candidate:
                return candidate
    return {}


def _extract_text(platform: str, payload: dict[str, Any], is_source: bool) -> str | None:
    if platform == "twitter":
        text = payload.get("full_text") or payload.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None
    if is_source:
        title = payload.get("title")
        selftext = payload.get("selftext")
        pieces = []
        if isinstance(title, str) and title.strip():
            pieces.append(title.strip())
        if isinstance(selftext, str) and selftext.strip():
            pieces.append(selftext.strip())
        return "\n\n".join(pieces) if pieces else None
    body = payload.get("body")
    return body.strip() if isinstance(body, str) and body.strip() else None


def _parse_twitter_time(value: Any) -> tuple[str | None, Any | None]:
    if not isinstance(value, str) or not value.strip():
        return None, None
    try:
        parsed = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), None
    except ValueError:
        return None, value


def _parse_unix_time(value: Any) -> tuple[str | None, Any | None]:
    if value is None:
        return None, None
    try:
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return parsed.isoformat().replace("+00:00", "Z"), None
    except (TypeError, ValueError, OSError):
        return None, value


def parse_created_at(platform: str, payload: dict[str, Any]) -> tuple[str | None, Any | None]:
    if platform == "twitter":
        return _parse_twitter_time(payload.get("created_at"))
    value = payload.get("created_utc") if payload.get("created_utc") is not None else payload.get("created")
    return _parse_unix_time(value)


def canonical_post_id(platform: str, raw_post_id: str) -> str:
    return f"RumourEval:post:{platform}:{raw_post_id}"


def canonical_user_id(platform: str, raw_user_id: str) -> str:
    return f"RumourEval:user:{platform}:{raw_user_id}"


def canonical_event_id(platform: str, split_name: str, thread_id: str) -> str:
    return f"RumourEval:event:{platform}:{split_name}:{thread_id}"


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    normalized = parsed._replace(
        fragment="",
        query=urlencode(query_pairs, doseq=True),
        netloc=parsed.netloc.lower(),
    )
    return urlunparse(normalized)


def _iter_visible_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and not is_hidden_artifact(path))


def _iter_visible_jsons(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.json") if path.is_file() and not is_hidden_artifact(path))


def _read_text_lines(path: Path) -> list[str]:
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return []


def _twitter_refs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    extended = payload.get("extended_entities") if isinstance(payload.get("extended_entities"), dict) else {}
    rows: list[dict[str, Any]] = []
    for item in entities.get("urls") or []:
        if isinstance(item, dict):
            raw_value = item.get("expanded_url") or item.get("url")
            if isinstance(raw_value, str) and raw_value.strip():
                rows.append({"ref_type": "url", "raw_value": raw_value, "extra": {"source": "entities.urls"}})
    media_items = []
    media_items.extend(entities.get("media") or [])
    media_items.extend(extended.get("media") or [])
    for item in media_items:
        if isinstance(item, dict):
            raw_value = item.get("media_url_https") or item.get("media_url") or item.get("expanded_url")
            if isinstance(raw_value, str) and raw_value.strip():
                rows.append({"ref_type": "media", "raw_value": raw_value, "extra": {"source": "entities.media"}})
    return rows


def _reddit_refs(payload: dict[str, Any], text_raw: str | None, is_source: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if is_source:
        raw_url = payload.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            rows.append({"ref_type": "url", "raw_value": raw_url, "extra": {"source": "payload.url"}})
        preview = payload.get("preview")
        if isinstance(preview, dict):
            for image in preview.get("images") or []:
                if isinstance(image, dict) and isinstance(image.get("source"), dict):
                    raw_value = image["source"].get("url")
                    if isinstance(raw_value, str) and raw_value.strip():
                        rows.append({"ref_type": "media", "raw_value": raw_value, "extra": {"source": "preview.images"}})
    if text_raw:
        for match in URL_PATTERN.findall(text_raw):
            rows.append({"ref_type": "url", "raw_value": match, "extra": {"source": "text_raw"}})
    return rows


def _collect_raw_tree_parent_map(raw_payload: Any) -> dict[str, str | None]:
    parent_map: dict[str, str | None] = {}

    def walk(node: Any, parent_raw_id: str | None = None) -> None:
        if isinstance(node, dict):
            kind = node.get("kind")
            data = node.get("data")
            if kind in {"t1", "t3"} and isinstance(data, dict):
                raw_id = str(data.get("id") or "").strip()
                if raw_id:
                    parent_map[raw_id] = parent_raw_id
                replies = data.get("replies")
                if isinstance(replies, (dict, list)):
                    walk(replies, raw_id or parent_raw_id)
                return
            for value in node.values():
                walk(value, parent_raw_id)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_raw_id)

    walk(raw_payload, None)
    return parent_map


def _collect_structure_parent_map(structure: Any, parent_raw_id: str | None = None) -> dict[str, str | None]:
    parent_map: dict[str, str | None] = {}
    if isinstance(structure, dict):
        for raw_id, child in structure.items():
            raw_id_str = str(raw_id).strip()
            if raw_id_str:
                parent_map.setdefault(raw_id_str, parent_raw_id)
            parent_map.update(_collect_structure_parent_map(child, raw_id_str))
    elif isinstance(structure, list):
        for item in structure:
            parent_map.update(_collect_structure_parent_map(item, parent_raw_id))
    return parent_map


class RumourEvalCanonicalConverter:
    dataset_id = "RumourEval"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.input_root = input_root
        self.output_root = Path(output_root)
        self.events: list[EventRecord] = []
        self.posts_by_id: dict[str, PostRecord] = {}
        self.users_by_id: dict[str, UserRecord] = {}
        self.references: list[ReferenceRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.thread_index: dict[str, dict[str, str]] = {}
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []
        self.skipped_items: list[dict[str, Any]] = []

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="twitter+reddit",
            task_family="rumour_veracity_and_stance",
            time_format="iso_8601_utc",
            label_spaces={
                "veracity": ["true", "false", "unverified"],
                "stance": ["support", "deny", "query", "comment"],
            },
            notes=(
                "Canonical conversion of RumourEval. One event equals one "
                "source post thread. Only explicit observable information and "
                "gold annotations are retained."
            ),
        )

        self._convert_threads()
        self._convert_annotations()
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_threads(self) -> None:
        roots = [
            (self.input_root / "rumoureval-2019-training-data" / "twitter-english", "twitter", "train", "train"),
            (self.input_root / "rumoureval-2019-training-data" / "reddit-training-data", "reddit", "train", "train"),
            (self.input_root / "rumoureval-2019-training-data" / "reddit-dev-data", "reddit", "train", "dev"),
            (self.input_root / "rumoureval-2019-test-data" / "twitter-en-test-data", "twitter", "test", "test"),
            (self.input_root / "rumoureval-2019-test-data" / "reddit-test-data", "reddit", "test", "test"),
        ]
        for root, platform, split_name, raw_split_source in roots:
            if not root.exists():
                self.validation_warnings.append(f"Missing expected root: {root}")
                continue
            if platform == "twitter":
                for topic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                    for thread_dir in sorted(path for path in topic_dir.iterdir() if path.is_dir()):
                        self._convert_thread(thread_dir, platform, split_name, raw_split_source, topic_dir.name)
            else:
                for thread_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                    self._convert_thread(thread_dir, platform, split_name, raw_split_source, None)

    def _convert_thread(
        self,
        thread_dir: Path,
        platform: str,
        split_name: str,
        raw_split_source: str,
        topic: str | None,
    ) -> None:
        source_paths = sorted((thread_dir / "source-tweet").glob("*.json"))
        if not source_paths:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "missing source post"})
            return

        source_payload = safe_read_json(source_paths[0])
        if platform == "reddit":
            source_payload = _unwrap_reddit_payload(source_payload)
        if not isinstance(source_payload, dict) or not source_payload:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "unreadable source post"})
            return

        thread_raw_id = thread_dir.name
        source_raw_id = str(source_payload.get("id_str") or source_payload.get("id") or thread_raw_id).strip()
        if not source_raw_id:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "missing source id"})
            return

        event_id = canonical_event_id(platform, split_name, thread_raw_id)
        source_post_id = canonical_post_id(platform, source_raw_id)
        replies: list[dict[str, Any]] = []
        for reply_path in sorted((thread_dir / "replies").glob("*.json")):
            payload = safe_read_json(reply_path)
            if platform == "reddit":
                payload = _unwrap_reddit_payload(payload)
            if isinstance(payload, dict) and payload:
                replies.append(payload)

        raw_tree = safe_read_json(thread_dir / "raw.json") if (thread_dir / "raw.json").exists() else {}
        structure = safe_read_json(thread_dir / "structure.json") if (thread_dir / "structure.json").exists() else {}
        raw_parent_map = _collect_raw_tree_parent_map(raw_tree)
        structure_parent_map = _collect_structure_parent_map(structure, source_raw_id)

        created_times: list[str] = []
        source_post = self._build_post(source_payload, platform, event_id, source_post_id, "source", None)
        if source_post.engagement.get("reply_count") is None:
            source_post.engagement["reply_count"] = len(replies)
        self.posts_by_id[source_post.post_id] = source_post
        if source_post.created_at:
            created_times.append(source_post.created_at)

        for payload in replies:
            raw_reply_id = str(payload.get("id_str") or payload.get("id") or "").strip()
            if not raw_reply_id:
                self.skipped_items.append({"type": "reply", "path": str(thread_dir), "reason": "missing reply id"})
                continue
            parent_raw_id = self._normalize_parent_raw_id(
                payload.get("in_reply_to_status_id_str")
                or payload.get("in_reply_to_status_id")
                or payload.get("parent_id")
            )
            parent_inferred = False
            if not parent_raw_id:
                parent_raw_id = raw_parent_map.get(raw_reply_id) or structure_parent_map.get(raw_reply_id)
                parent_inferred = parent_raw_id is not None
            if not parent_raw_id:
                parent_raw_id = source_raw_id
                parent_inferred = True
            parent_post_id = canonical_post_id(platform, parent_raw_id)
            reply_post = self._build_post(payload, platform, event_id, source_post_id, "response", parent_post_id)
            if parent_inferred:
                reply_post.extra["parent_inferred"] = True
            self.posts_by_id[reply_post.post_id] = reply_post
            if reply_post.created_at:
                created_times.append(reply_post.created_at)

        self.events.append(
            EventRecord(
                event_id=event_id,
                dataset_id=self.dataset_id,
                platform=platform,
                source_post_id=source_post_id,
                split=split_name,
                topic=topic,
                lang="en",
                start_time=min(created_times) if created_times else None,
                end_time=max(created_times) if created_times else None,
                view_status={
                    "semantic": "available",
                    "stance": "text_only",
                    "propagation": "full_tree" if structure else "flat",
                    "evidence": "none",
                    "source": "partial",
                },
                extra={
                    "thread_path": str(thread_dir),
                    "raw_split_source": raw_split_source,
                    "has_raw_json": (thread_dir / "raw.json").exists(),
                    "topic_dir": topic,
                },
            )
        )
        self.thread_index[source_raw_id] = {
            "event_id": event_id,
            "source_post_id": source_post_id,
        }
        self.thread_index[thread_raw_id] = dict(self.thread_index[source_raw_id])

    def _build_post(
        self,
        payload: dict[str, Any],
        platform: str,
        event_id: str,
        root_post_id: str,
        post_type: str,
        parent_post_id: str | None,
    ) -> PostRecord:
        raw_post_id = str(payload.get("id_str") or payload.get("id") or "").strip()
        post_id = canonical_post_id(platform, raw_post_id)
        user_id = self._upsert_user(platform, payload)
        created_at, created_raw = parse_created_at(platform, payload)
        text_raw = _extract_text(platform, payload, is_source=(post_type == "source"))

        like_count = self._as_int(payload.get("favorite_count")) if platform == "twitter" else self._as_int(payload.get("score"))
        reply_count = self._as_int(payload.get("reply_count")) if platform == "twitter" else None
        repost_count = self._as_int(payload.get("retweet_count")) if platform == "twitter" else None

        refs = self._extract_references(platform, payload, text_raw, event_id, post_id, post_type == "source")
        self.references.extend(refs)

        extra: dict[str, Any] = {}
        if created_raw is not None:
            extra["created_at_raw"] = created_raw
        if payload.get("_reddit_kind") == "more":
            extra["reddit_kind"] = "more"
        for key in ["subreddit", "domain", "permalink", "conversation_id", "possibly_sensitive", "author_flair_text", "thumbnail", "is_video"]:
            if key in payload:
                extra[key] = payload.get(key)

        return PostRecord(
            post_id=post_id,
            event_id=event_id,
            user_id=user_id,
            post_type=post_type,
            interaction_type="source" if post_type == "source" else "reply",
            text_status=classify_text_status(
                text_raw,
                missing_policy="unknown" if payload.get("_reddit_kind") == "more" else "empty",
            ) if post_type != "source" else classify_text_status(text_raw, missing_policy="unknown"),
            parent_post_id=parent_post_id,
            root_post_id=root_post_id,
            created_at=created_at,
            text_raw=text_raw,
            lang="en",
            engagement={
                "like_count": like_count,
                "reply_count": reply_count,
                "repost_count": repost_count,
            },
            has_media=any(ref.ref_type == "media" for ref in refs),
            has_url=any(ref.ref_type == "url" for ref in refs),
            extra=extra,
        )

    def _extract_references(
        self,
        platform: str,
        payload: dict[str, Any],
        text_raw: str | None,
        event_id: str,
        post_id: str,
        is_source: bool,
    ) -> list[ReferenceRecord]:
        raw_refs = _twitter_refs(payload) if platform == "twitter" else _reddit_refs(payload, text_raw, is_source)
        rows: list[ReferenceRecord] = []
        seen: set[tuple[str, str]] = set()
        for ref in raw_refs:
            raw_value = ref["raw_value"]
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            key = (ref["ref_type"], raw_value.strip())
            if key in seen:
                continue
            seen.add(key)
            normalized_value = normalize_url(raw_value)
            rows.append(
                ReferenceRecord(
                    ref_id=f"RumourEval:ref:{post_id}:{len(rows)}",
                    event_id=event_id,
                    post_id=post_id,
                    ref_type=ref["ref_type"],
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    domain=urlparse(normalized_value).netloc.lower() or None,
                    target_post_id=None,
                    extra=ref.get("extra", {}),
                )
            )
        return rows

    def _upsert_user(self, platform: str, payload: dict[str, Any]) -> str | None:
        if platform == "twitter":
            user_payload = payload.get("user") if isinstance(payload.get("user"), dict) else {}
            raw_user_id = str(
                user_payload.get("id_str")
                or user_payload.get("id")
                or user_payload.get("screen_name")
                or ""
            ).strip()
            if not raw_user_id:
                return None
            user_id = canonical_user_id(platform, raw_user_id)
            if user_id not in self.users_by_id:
                account_created_at, raw_created = parse_created_at(platform, user_payload)
                extra: dict[str, Any] = {}
                for key in ["screen_name", "listed_count"]:
                    if key in user_payload:
                        extra[key] = user_payload.get(key)
                if raw_created is not None:
                    extra["account_created_at_raw"] = raw_created
                self.users_by_id[user_id] = UserRecord(
                    user_id=user_id,
                    platform=platform,
                    account_created_at=account_created_at,
                    verified=self._as_bool_or_none(user_payload.get("verified")),
                    followers_count=self._as_int(user_payload.get("followers_count")),
                    following_count=self._as_int(user_payload.get("friends_count")),
                    post_count=self._as_int(user_payload.get("statuses_count")),
                    profile_description_raw=self._as_text_or_none(user_payload.get("description")),
                    display_name_raw=self._as_text_or_none(user_payload.get("name")),
                    extra=extra,
                )
            return user_id

        raw_author = (str(payload.get("author") or "[deleted]").strip() or "[deleted]")
        user_id = canonical_user_id(platform, raw_author)
        if user_id not in self.users_by_id:
            extra: dict[str, Any] = {}
            for key in ["subreddit", "author_flair_text", "author_flair_css_class"]:
                if key in payload:
                    extra[key] = payload.get(key)
            self.users_by_id[user_id] = UserRecord(
                user_id=user_id,
                platform=platform,
                account_created_at=None,
                verified=None,
                followers_count=None,
                following_count=None,
                post_count=None,
                profile_description_raw=None,
                display_name_raw=raw_author,
                extra=extra,
            )
        return user_id

    def _convert_annotations(self) -> None:
        files = [
            (self.input_root / "rumoureval-2019-training-data" / "train-key.json", "train-key.json"),
            (self.input_root / "rumoureval-2019-training-data" / "dev-key.json", "dev-key.json"),
            (self.input_root / "rumoureval-2019-test-data" / "test-key.json", "test-key.json"),
        ]
        for path, file_name in files:
            if not path.exists():
                self.validation_warnings.append(f"Missing annotation file: {path}")
                continue
            payload = safe_read_json(path)
            if not isinstance(payload, dict):
                self.validation_warnings.append(f"Unreadable annotation file: {path}")
                continue

            stance_map = payload.get("subtaskaenglish") or {}
            if isinstance(stance_map, dict):
                for raw_post_id, raw_label in stance_map.items():
                    label = str(raw_label).strip().lower()
                    if label not in STANCE_LABELS:
                        continue
                    target_post_id = self._find_post_id_by_raw_id(str(raw_post_id))
                    if target_post_id is None:
                        self.validation_warnings.append(f"Missing stance target post for raw id {raw_post_id}")
                        continue
                    anchor_post_id = self.posts_by_id[target_post_id].root_post_id
                    self.annotations.append(
                        AnnotationRecord(
                            annotation_id=f"RumourEval:ann:stance:{target_post_id}",
                            target_type="post",
                            target_id=target_post_id,
                            task="stance",
                            label=label,
                            label_space="stance",
                            original_label=str(raw_label),
                            anchor_post_id=anchor_post_id,
                            annotator_kind="gold",
                            confidence=None,
                            extra={
                                "source_file": file_name,
                                "source_table": "subtaskaenglish",
                                "raw_post_id": str(raw_post_id),
                            },
                        )
                    )

            veracity_map = payload.get("subtaskbenglish") or {}
            if isinstance(veracity_map, dict):
                for raw_event_id, raw_label in veracity_map.items():
                    label = str(raw_label).strip().lower()
                    if label not in VERACITY_LABELS:
                        continue
                    event_ref = self.thread_index.get(str(raw_event_id))
                    if event_ref is None:
                        self.validation_warnings.append(f"Missing veracity target event for raw id {raw_event_id}")
                        continue
                    self.annotations.append(
                        AnnotationRecord(
                            annotation_id=f"RumourEval:ann:veracity:{event_ref['event_id']}",
                            target_type="event",
                            target_id=event_ref["event_id"],
                            task="veracity",
                            label=label,
                            label_space="veracity",
                            original_label=str(raw_label),
                            anchor_post_id=event_ref["source_post_id"],
                            annotator_kind="gold",
                            confidence=None,
                            extra={
                                "source_file": file_name,
                                "source_table": "subtaskbenglish",
                                "raw_event_id": str(raw_event_id),
                            },
                        )
                    )

        stance_anchor_ids = {
            annotation.anchor_post_id
            for annotation in self.annotations
            if annotation.task == "stance" and annotation.anchor_post_id
        }
        for event in self.events:
            if event.source_post_id in stance_anchor_ids:
                event.view_status["stance"] = "gold"

    def _find_post_id_by_raw_id(self, raw_post_id: str) -> str | None:
        for platform in ["twitter", "reddit"]:
            post_id = canonical_post_id(platform, raw_post_id)
            if post_id in self.posts_by_id:
                return post_id
        return None

    def _validate(self) -> dict[str, Any]:
        event_map = {event.event_id: event for event in self.events}
        posts_by_event: dict[str, list[PostRecord]] = defaultdict(list)
        for post in self.posts_by_id.values():
            posts_by_event[post.event_id].append(post)

        for event in self.events:
            source_posts = [post for post in posts_by_event[event.event_id] if post.post_type == "source"]
            if len(source_posts) != 1:
                self.validation_errors.append(
                    f"Event {event.event_id} must have exactly one source post, found {len(source_posts)}"
                )
                continue
            source_post = source_posts[0]
            if source_post.post_id != event.source_post_id:
                self.validation_errors.append(
                    f"Event {event.event_id} source_post_id mismatch: {event.source_post_id} vs {source_post.post_id}"
                )
            if source_post.parent_post_id is not None:
                self.validation_errors.append(f"Source post {source_post.post_id} has non-null parent_post_id")
            if source_post.root_post_id != source_post.post_id:
                self.validation_errors.append(f"Source post {source_post.post_id} root_post_id must equal post_id")
            root_ids = {post.root_post_id for post in posts_by_event[event.event_id]}
            if root_ids != {event.source_post_id}:
                self.validation_errors.append(
                    f"Event {event.event_id} posts do not share a single root_post_id: {sorted(root_ids)}"
                )

        for annotation in self.annotations:
            if annotation.target_type == "event" and annotation.target_id not in event_map:
                self.validation_errors.append(f"Missing event annotation target: {annotation.target_id}")
            if annotation.target_type == "post" and annotation.target_id not in self.posts_by_id:
                self.validation_errors.append(f"Missing post annotation target: {annotation.target_id}")
        for post in self.posts_by_id.values():
            validate_post_schema(post, self.validation_errors)

        split_counts = {"train": 0, "test": 0}
        platform_counts = {"twitter": 0, "reddit": 0}
        view_coverage: dict[str, dict[str, int]] = {
            key: defaultdict(int) for key in ["semantic", "stance", "propagation", "evidence", "source"]
        }
        annotations_by_task = Counter(annotation.task for annotation in self.annotations)
        event_ids_with_references = {reference.event_id for reference in self.references}
        post_ids_with_references = {reference.post_id for reference in self.references}
        posts_with_url = {post.post_id for post in self.posts_by_id.values() if post.has_url}
        posts_with_media = {post.post_id for post in self.posts_by_id.values() if post.has_media}
        for event in self.events:
            split_counts[event.split] = split_counts.get(event.split, 0) + 1
            platform_counts[event.platform] = platform_counts.get(event.platform, 0) + 1
            for key, value in event.view_status.items():
                view_coverage[key][value] += 1

        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "events": len(self.events),
                "posts": len(self.posts_by_id),
                "users": len(self.users_by_id),
                "references": len(self.references),
                "annotations": len(self.annotations),
            },
            "split_counts": split_counts,
            "platform_counts": platform_counts,
            "annotation_coverage": {
                "events_with_veracity_labels": len({ann.target_id for ann in self.annotations if ann.task == "veracity"}),
                "posts_with_stance_labels": len({ann.target_id for ann in self.annotations if ann.task == "stance"}),
            },
            "annotations_by_task": dict(annotations_by_task),
            "local_raw_annotation_counts": self._collect_local_raw_annotation_counts(),
            "reference_coverage": {
                "events_with_references": len(event_ids_with_references),
                "posts_with_references": len(post_ids_with_references),
                "posts_with_url": len(posts_with_url),
                "posts_with_media": len(posts_with_media),
            },
            "view_coverage": {key: dict(value) for key, value in view_coverage.items()},
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skipped_items": self.skipped_items,
            "schema_valid": len(self.validation_errors) == 0,
        }

    def _write_outputs(self, meta: DatasetMeta, report: dict[str, Any]) -> None:
        write_json(self.output_root / "dataset_meta.json", meta.to_dict())
        write_jsonl(self.output_root / "events.jsonl", [row.to_dict() for row in self.events])
        write_jsonl(self.output_root / "posts.jsonl", [row.to_dict() for row in self.posts_by_id.values()])
        write_jsonl(self.output_root / "users.jsonl", [row.to_dict() for row in self.users_by_id.values()])
        write_jsonl(self.output_root / "references.jsonl", [row.to_dict() for row in self.references])
        write_jsonl(self.output_root / "annotations.jsonl", [row.to_dict() for row in self.annotations])
        write_json(self.output_root / "validation_report.json", report)

    @staticmethod
    def _normalize_parent_raw_id(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.startswith("t1_") or text.startswith("t3_"):
            text = text.split("_", 1)[1]
        return text

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_bool_or_none(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        return None

    @staticmethod
    def _as_text_or_none(value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    def _collect_local_raw_annotation_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        files = [
            ("train", self.input_root / "rumoureval-2019-training-data" / "train-key.json"),
            ("dev", self.input_root / "rumoureval-2019-training-data" / "dev-key.json"),
            ("test", self.input_root / "rumoureval-2019-test-data" / "test-key.json"),
        ]
        for split_name, path in files:
            if not path.exists():
                continue
            payload = safe_read_json(path)
            if not isinstance(payload, dict):
                continue
            counts[split_name] = {
                "subtaskaenglish": len(payload.get("subtaskaenglish") or {}),
                "subtaskbenglish": len(payload.get("subtaskbenglish") or {}),
            }
        return counts


class PhemeVeracityCanonicalConverter:
    dataset_id = "PHEME"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.input_root = input_root
        self.output_root = Path(output_root)
        self.events: list[EventRecord] = []
        self.posts_by_id: dict[str, PostRecord] = {}
        self.users_by_id: dict[str, UserRecord] = {}
        self.references: list[ReferenceRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []
        self.skipped_items: list[dict[str, Any]] = []

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="twitter",
            task_family="rumour_veracity_and_thread_structure",
            time_format="iso_8601_utc",
            label_spaces={
                "rumour_status": ["rumour", "non-rumour", "unclear"],
                "veracity": ["true", "false", "unverified", "non-rumour"],
            },
            notes=(
                "Canonical conversion of the PHEME veracity dataset. One thread "
                "equals one event. Only canonicalized raw observable information "
                "and explicit or mapped annotations are retained."
            ),
        )

        self._convert_threads()
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_threads(self) -> None:
        root = self.input_root / "all-rnr-annotated-threads"
        if not root.exists():
            raise FileNotFoundError(f"Missing expected PHEME root: {root}")

        for topic_dir in _iter_visible_dirs(root):
            for group_name in ["rumours", "non-rumours"]:
                group_dir = topic_dir / group_name
                if not group_dir.exists():
                    continue
                for thread_dir in _iter_visible_dirs(group_dir):
                    self._convert_thread(topic_dir.name, group_name, thread_dir)

    def _convert_thread(self, topic: str, group_name: str, thread_dir: Path) -> None:
        source_paths = _iter_visible_jsons(thread_dir / "source-tweets")
        if not source_paths:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "missing source tweet"})
            return

        source_payload = safe_read_json(source_paths[0])
        if not isinstance(source_payload, dict) or not source_payload:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "unreadable source tweet"})
            return

        source_raw_id = str(source_payload.get("id_str") or source_payload.get("id") or thread_dir.name).strip()
        if not source_raw_id:
            self.skipped_items.append({"type": "thread", "path": str(thread_dir), "reason": "missing source id"})
            return

        event_id = self._canonical_event_id(topic, group_name, thread_dir.name)
        source_post_id = self._canonical_post_id(source_raw_id)

        structure = safe_read_json(thread_dir / "structure.json") if (thread_dir / "structure.json").exists() else {}
        parent_map = _collect_structure_parent_map(structure, source_raw_id)

        created_times: list[str] = []
        event_refs: list[ReferenceRecord] = []

        source_post, source_refs = self._build_post(
            payload=source_payload,
            event_id=event_id,
            root_post_id=source_post_id,
            post_type="source",
            parent_post_id=None,
        )
        self.posts_by_id[source_post.post_id] = source_post
        event_refs.extend(source_refs)
        if source_post.created_at:
            created_times.append(source_post.created_at)

        reactions = []
        for reaction_path in _iter_visible_jsons(thread_dir / "reactions"):
            payload = safe_read_json(reaction_path)
            if isinstance(payload, dict) and payload:
                reactions.append(payload)

        if source_post.engagement.get("reply_count") is None:
            source_post.engagement["reply_count"] = len(reactions)

        star_tree = True
        for payload in reactions:
            raw_reply_id = str(payload.get("id_str") or payload.get("id") or "").strip()
            if not raw_reply_id:
                self.skipped_items.append({"type": "reaction", "path": str(thread_dir), "reason": "missing reaction id"})
                continue
            if raw_reply_id == source_raw_id:
                self.skipped_items.append(
                    {"type": "reaction", "path": str(thread_dir), "reason": "duplicate source tweet inside reactions"}
                )
                continue
            parent_raw_id = RumourEvalCanonicalConverter._normalize_parent_raw_id(
                payload.get("in_reply_to_status_id_str") or payload.get("in_reply_to_status_id")
            )
            parent_inferred = False
            if not parent_raw_id:
                parent_raw_id = parent_map.get(raw_reply_id)
                parent_inferred = parent_raw_id is not None
            if not parent_raw_id:
                parent_raw_id = source_raw_id
                parent_inferred = True
            if parent_raw_id != source_raw_id:
                star_tree = False
            parent_post_id = self._canonical_post_id(parent_raw_id)

            reply_post, reply_refs = self._build_post(
                payload=payload,
                event_id=event_id,
                root_post_id=source_post_id,
                post_type="response",
                parent_post_id=parent_post_id,
            )
            if parent_inferred:
                reply_post.extra["parent_inferred"] = True
            self.posts_by_id[reply_post.post_id] = reply_post
            event_refs.extend(reply_refs)
            if reply_post.created_at:
                created_times.append(reply_post.created_at)

        self.references.extend(event_refs)
        annotation_payload = safe_read_json(thread_dir / "annotation.json") if (thread_dir / "annotation.json").exists() else {}
        annotation_payload = annotation_payload if isinstance(annotation_payload, dict) else {}
        self._append_annotations(
            event_id=event_id,
            source_post_id=source_post_id,
            topic=topic,
            group_name=group_name,
            thread_dir=thread_dir,
            annotation_payload=annotation_payload,
        )

        source_lang = RumourEvalCanonicalConverter._as_text_or_none(source_payload.get("lang"))
        evidence_view = "explicit_refs" if any(ref.ref_type == "url" for ref in event_refs) else "none"
        if reactions:
            propagation_view = "star" if star_tree else "full_tree"
        else:
            propagation_view = "flat"

        self.events.append(
            EventRecord(
                event_id=event_id,
                dataset_id=self.dataset_id,
                platform="twitter",
                source_post_id=source_post_id,
                split="all",
                topic=topic,
                lang=source_lang,
                start_time=min(created_times) if created_times else None,
                end_time=max(created_times) if created_times else None,
                view_status={
                    "semantic": "available",
                    "stance": "text_only" if reactions else "none",
                    "propagation": propagation_view,
                    "evidence": evidence_view,
                    "source": "partial",
                },
                extra={
                    "thread_path": str(thread_dir),
                    "thread_group": group_name,
                    "annotation_path": str(thread_dir / "annotation.json") if (thread_dir / "annotation.json").exists() else None,
                    "structure_path": str(thread_dir / "structure.json") if (thread_dir / "structure.json").exists() else None,
                },
            )
        )

    def _append_annotations(
        self,
        event_id: str,
        source_post_id: str,
        topic: str,
        group_name: str,
        thread_dir: Path,
        annotation_payload: dict[str, Any],
    ) -> None:
        raw_rumour_status = str(annotation_payload.get("is_rumour") or "").strip().lower()
        if raw_rumour_status == "nonrumour":
            rumour_status = "non-rumour"
        elif raw_rumour_status == "rumour":
            rumour_status = "rumour"
        elif raw_rumour_status == "unclear":
            rumour_status = "unclear"
        else:
            rumour_status = None
        if rumour_status in PHEME_RUMOUR_STATUS_LABELS:
            self.annotations.append(
                AnnotationRecord(
                    annotation_id=f"{self.dataset_id}:ann:rumour_status:{event_id}",
                    target_type="event",
                    target_id=event_id,
                    task="rumour_status",
                    label=rumour_status,
                    label_space="rumour_status",
                    original_label=raw_rumour_status,
                    anchor_post_id=source_post_id,
                    annotator_kind="gold",
                    confidence=None,
                    extra={
                        "source_file": "annotation.json",
                        "annotation_path": str(thread_dir / "annotation.json"),
                        "topic": topic,
                        "thread_group": group_name,
                    },
                )
            )

        mapped_veracity = self._map_pheme_veracity(annotation_payload)
        if mapped_veracity is None and group_name == "non-rumours":
            mapped_veracity = "non-rumour"

        if mapped_veracity in PHEME_VERACITY_LABELS:
            self.annotations.append(
                AnnotationRecord(
                    annotation_id=f"{self.dataset_id}:ann:veracity:{event_id}",
                    target_type="event",
                    target_id=event_id,
                    task="veracity",
                    label=mapped_veracity,
                    label_space="veracity",
                    original_label=self._stringify_original_veracity(annotation_payload),
                    anchor_post_id=source_post_id,
                    annotator_kind="mapped",
                    confidence=None,
                    extra={
                        "source_file": "annotation.json",
                        "annotation_path": str(thread_dir / "annotation.json"),
                        "topic": topic,
                        "thread_group": group_name,
                        "category": annotation_payload.get("category"),
                        "links": annotation_payload.get("links"),
                        "is_turnaround": annotation_payload.get("is_turnaround"),
                        "misinformation": annotation_payload.get("misinformation"),
                        "true": annotation_payload.get("true"),
                    },
                )
            )

    def _build_post(
        self,
        payload: dict[str, Any],
        event_id: str,
        root_post_id: str,
        post_type: str,
        parent_post_id: str | None,
    ) -> tuple[PostRecord, list[ReferenceRecord]]:
        raw_post_id = str(payload.get("id_str") or payload.get("id") or "").strip()
        post_id = self._canonical_post_id(raw_post_id)
        user_id = self._upsert_user(payload)
        created_at, created_raw = parse_created_at("twitter", payload)
        text_raw = _extract_text("twitter", payload, is_source=(post_type == "source"))

        refs = self._extract_references(payload, event_id, post_id)
        extra: dict[str, Any] = {}
        if created_raw is not None:
            extra["created_at_raw"] = created_raw
        for key in ["source", "conversation_id", "possibly_sensitive", "coordinates", "geo", "place", "truncated"]:
            if key in payload:
                extra[key] = payload.get(key)

        post = PostRecord(
            post_id=post_id,
            event_id=event_id,
            user_id=user_id,
            post_type=post_type,
            interaction_type="source" if post_type == "source" else "reply",
            text_status=classify_text_status(
                text_raw,
                missing_policy="empty" if post_type != "source" else "unknown",
            ),
            parent_post_id=parent_post_id,
            root_post_id=root_post_id,
            created_at=created_at,
            text_raw=text_raw,
            lang=RumourEvalCanonicalConverter._as_text_or_none(payload.get("lang")),
            engagement={
                "like_count": RumourEvalCanonicalConverter._as_int(payload.get("favorite_count")),
                "reply_count": None,
                "repost_count": RumourEvalCanonicalConverter._as_int(payload.get("retweet_count")),
            },
            has_media=any(ref.ref_type == "media" for ref in refs),
            has_url=any(ref.ref_type == "url" for ref in refs),
            extra=extra,
        )
        return post, refs

    def _extract_references(self, payload: dict[str, Any], event_id: str, post_id: str) -> list[ReferenceRecord]:
        rows: list[ReferenceRecord] = []
        seen: set[tuple[str, str]] = set()
        for ref in _twitter_refs(payload):
            raw_value = ref["raw_value"]
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            key = (ref["ref_type"], raw_value.strip())
            if key in seen:
                continue
            seen.add(key)
            normalized_value = normalize_url(raw_value)
            rows.append(
                ReferenceRecord(
                    ref_id=f"{self.dataset_id}:ref:{post_id}:{len(rows)}",
                    event_id=event_id,
                    post_id=post_id,
                    ref_type=ref["ref_type"],
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    domain=urlparse(normalized_value).netloc.lower() or None,
                    target_post_id=None,
                    extra=ref.get("extra") or {},
                )
            )
        return rows

    def _upsert_user(self, payload: dict[str, Any]) -> str | None:
        user_payload = payload.get("user")
        if isinstance(user_payload, dict):
            raw_user_id = str(
                user_payload.get("id_str")
                or user_payload.get("id")
                or user_payload.get("screen_name")
                or user_payload.get("name")
                or "unknown"
            ).strip()
            user_id = self._canonical_user_id(raw_user_id)
            if user_id not in self.users_by_id:
                account_created_at, created_raw = parse_created_at("twitter", user_payload)
                extra: dict[str, Any] = {}
                for key in ["screen_name", "listed_count", "location", "url"]:
                    if key in user_payload:
                        extra[key] = user_payload.get(key)
                if created_raw is not None:
                    extra["account_created_at_raw"] = created_raw
                self.users_by_id[user_id] = UserRecord(
                    user_id=user_id,
                    platform="twitter",
                    account_created_at=account_created_at,
                    verified=RumourEvalCanonicalConverter._as_bool_or_none(user_payload.get("verified")),
                    followers_count=RumourEvalCanonicalConverter._as_int(user_payload.get("followers_count")),
                    following_count=RumourEvalCanonicalConverter._as_int(user_payload.get("friends_count")),
                    post_count=RumourEvalCanonicalConverter._as_int(user_payload.get("statuses_count")),
                    profile_description_raw=RumourEvalCanonicalConverter._as_text_or_none(user_payload.get("description")),
                    display_name_raw=RumourEvalCanonicalConverter._as_text_or_none(user_payload.get("name")),
                    extra=extra,
                )
            return user_id
        return None

    def _validate(self) -> dict[str, Any]:
        event_map = {event.event_id: event for event in self.events}
        posts_by_event: dict[str, list[PostRecord]] = defaultdict(list)
        for post in self.posts_by_id.values():
            posts_by_event[post.event_id].append(post)

        for event in self.events:
            source_posts = [post for post in posts_by_event[event.event_id] if post.post_type == "source"]
            if len(source_posts) != 1:
                self.validation_errors.append(
                    f"Event {event.event_id} must have exactly one source post, found {len(source_posts)}"
                )
                continue
            source_post = source_posts[0]
            if source_post.post_id != event.source_post_id:
                self.validation_errors.append(
                    f"Event {event.event_id} source_post_id mismatch: {event.source_post_id} vs {source_post.post_id}"
                )
            if source_post.parent_post_id is not None:
                self.validation_errors.append(f"Source post {source_post.post_id} has non-null parent_post_id")
            if source_post.root_post_id != source_post.post_id:
                self.validation_errors.append(f"Source post {source_post.post_id} root_post_id must equal post_id")
            root_ids = {post.root_post_id for post in posts_by_event[event.event_id]}
            if root_ids != {event.source_post_id}:
                self.validation_errors.append(
                    f"Event {event.event_id} posts do not share a single root_post_id: {sorted(root_ids)}"
                )

        for annotation in self.annotations:
            if annotation.target_type == "event" and annotation.target_id not in event_map:
                self.validation_errors.append(f"Missing event annotation target: {annotation.target_id}")
            if annotation.target_type == "post" and annotation.target_id not in self.posts_by_id:
                self.validation_errors.append(f"Missing post annotation target: {annotation.target_id}")
        for post in self.posts_by_id.values():
            validate_post_schema(post, self.validation_errors)

        split_counts: dict[str, int] = {"all": 0}
        view_coverage: dict[str, dict[str, int]] = {
            key: defaultdict(int) for key in ["semantic", "stance", "propagation", "evidence", "source"]
        }
        topic_counts: dict[str, int] = defaultdict(int)
        for event in self.events:
            split_counts[event.split] = split_counts.get(event.split, 0) + 1
            if event.topic:
                topic_counts[event.topic] += 1
            for key, value in event.view_status.items():
                view_coverage[key][value] += 1

        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "events": len(self.events),
                "posts": len(self.posts_by_id),
                "users": len(self.users_by_id),
                "references": len(self.references),
                "annotations": len(self.annotations),
            },
            "split_counts": split_counts,
            "platform_counts": {"twitter": len(self.events)},
            "topic_counts": dict(topic_counts),
            "annotation_coverage": {
                "events_with_rumour_status_labels": len(
                    {ann.target_id for ann in self.annotations if ann.task == "rumour_status"}
                ),
                "events_with_veracity_labels": len({ann.target_id for ann in self.annotations if ann.task == "veracity"}),
            },
            "view_coverage": {key: dict(value) for key, value in view_coverage.items()},
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skipped_items": self.skipped_items,
            "schema_valid": len(self.validation_errors) == 0,
        }

    def _write_outputs(self, meta: DatasetMeta, report: dict[str, Any]) -> None:
        write_json(self.output_root / "dataset_meta.json", meta.to_dict())
        write_jsonl(self.output_root / "events.jsonl", [row.to_dict() for row in self.events])
        write_jsonl(self.output_root / "posts.jsonl", [row.to_dict() for row in self.posts_by_id.values()])
        write_jsonl(self.output_root / "users.jsonl", [row.to_dict() for row in self.users_by_id.values()])
        write_jsonl(self.output_root / "references.jsonl", [row.to_dict() for row in self.references])
        write_jsonl(self.output_root / "annotations.jsonl", [row.to_dict() for row in self.annotations])
        write_json(self.output_root / "validation_report.json", report)

    @classmethod
    def _canonical_event_id(cls, topic: str, group_name: str, thread_id: str) -> str:
        return f"{cls.dataset_id}:event:{topic}:{group_name}:{thread_id}"

    @classmethod
    def _canonical_post_id(cls, raw_post_id: str) -> str:
        return f"{cls.dataset_id}:post:twitter:{raw_post_id}"

    @classmethod
    def _canonical_user_id(cls, raw_user_id: str) -> str:
        return f"{cls.dataset_id}:user:twitter:{raw_user_id}"

    @staticmethod
    def _stringify_original_veracity(annotation_payload: dict[str, Any]) -> str:
        pieces = []
        for key in ["is_rumour", "misinformation", "true"]:
            if key in annotation_payload:
                pieces.append(f"{key}={annotation_payload.get(key)}")
        return "; ".join(pieces) if pieces else "missing"

    @staticmethod
    def _map_pheme_veracity(annotation_payload: dict[str, Any]) -> str | None:
        if str(annotation_payload.get("is_rumour") or "").strip().lower() == "nonrumour":
            return "non-rumour"
        if "misinformation" in annotation_payload and "true" in annotation_payload:
            misinformation = str(annotation_payload.get("misinformation")).strip()
            truth = str(annotation_payload.get("true")).strip()
            if misinformation == "0" and truth == "0":
                return "unverified"
            if misinformation == "0" and truth == "1":
                return "true"
            if misinformation == "1" and truth == "0":
                return "false"
            return None
        if "misinformation" in annotation_payload and "true" not in annotation_payload:
            misinformation = str(annotation_payload.get("misinformation")).strip()
            if misinformation == "0":
                return "unverified"
            if misinformation == "1":
                return "false"
        return None


class RumorDetectionACL2017CanonicalConverter:
    dataset_id = "ACL2017_combined"
    delay_bucket_minutes = 0.5

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.input_root = input_root
        self.output_root = Path(output_root)
        self.events: list[EventRecord] = []
        self.posts_by_id: dict[str, PostRecord] = {}
        self.users_by_id: dict[str, UserRecord] = {}
        self.references: list[ReferenceRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []
        self.skipped_items: list[dict[str, Any]] = []

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="twitter",
            task_family="rumour_detection_and_propagation",
            time_format="iso_8601_utc",
            label_spaces={
                "rumour_status": ["rumour", "non-rumour"],
                "veracity": ["true", "false", "unverified", "non-rumour"],
            },
            notes=(
                "Canonical conversion of the ACL 2017 rumor detection datasets "
                "(twitter15 and twitter16). Source tweet text is available, "
                "while reply text and full user metadata are not included in the "
                "raw release. Tree nodes are preserved as observable structural "
                "posts with null text/timestamp where the raw release omits them."
            ),
        )

        self._convert_subsets()
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_subsets(self) -> None:
        for subset in ["twitter15", "twitter16"]:
            subset_root = self.input_root / subset
            if not subset_root.exists():
                self.validation_warnings.append(f"Missing expected subset root: {subset_root}")
                continue
            label_map = self._load_label_map(subset_root / "label.txt")
            source_map = self._load_source_map(subset_root / "source_tweets.txt")
            tree_root = subset_root / "tree"
            for tree_path in sorted(path for path in tree_root.glob("*.txt") if path.is_file()):
                source_tweet_id = tree_path.stem
                label = label_map.get(source_tweet_id)
                source_text = source_map.get(source_tweet_id)
                self._convert_thread(subset, tree_path, source_tweet_id, source_text, label)

    def _convert_thread(
        self,
        subset: str,
        tree_path: Path,
        source_tweet_id: str,
        source_text: str | None,
        raw_label: str | None,
    ) -> None:
        edges, root_node = self._parse_tree_edges(tree_path)
        if root_node is None:
            self.skipped_items.append({"type": "thread", "path": str(tree_path), "reason": "missing root node"})
            return

        event_id = self._canonical_event_id(subset, source_tweet_id)
        source_bucket = self._bucket_delay(root_node["delay"])
        source_node_key = self._post_key(root_node["tweet_id"], source_bucket)
        source_post_id = self._canonical_post_id(subset, root_node["tweet_id"], source_bucket)

        post_state: dict[tuple[str, float], dict[str, Any]] = {
            source_node_key: {
                "tweet_id_raw": root_node["tweet_id"],
                "delay_bucket_minutes": source_bucket,
                "raw_delay_values": {str(root_node["delay"])},
                "uid_candidates": [root_node["uid"]],
                "parent_candidates": Counter(),
                "is_source": True,
            }
        }

        for parent_node, child_node in edges:
            parent_key = self._post_key(parent_node["tweet_id"], self._bucket_delay(parent_node["delay"]))
            child_key = self._post_key(child_node["tweet_id"], self._bucket_delay(child_node["delay"]))
            self._register_node(post_state, parent_node, is_source=(parent_key == source_node_key))
            self._register_node(post_state, child_node, is_source=(child_key == source_node_key))
            if parent_key != child_key:
                post_state[child_key]["parent_candidates"][parent_key] += 1

        reply_counts = defaultdict(int)
        max_delay_minutes = 0.0
        sorted_keys = sorted(post_state.keys(), key=lambda item: (item[1], item[0]))
        for node_key in sorted_keys:
            state = post_state[node_key]
            tweet_id_raw, delay_bucket = node_key
            representative_uid = state["uid_candidates"][0] if state["uid_candidates"] else ""
            user_id = self._upsert_user(representative_uid, subset)
            post_id = self._canonical_post_id(subset, tweet_id_raw, delay_bucket)
            if state["is_source"]:
                parent_post_id = None
                post_type = "source"
                text_raw = source_text
                source_text_available = source_text is not None
            else:
                parent_key = self._choose_parent_key(state["parent_candidates"], source_node_key)
                parent_post_id = self._canonical_post_id(subset, parent_key[0], parent_key[1]) if parent_key else source_post_id
                post_type = "response"
                text_raw = None
                source_text_available = False
                reply_counts[parent_post_id] += 1
            max_delay_minutes = max(max_delay_minutes, delay_bucket)
            self.posts_by_id[post_id] = PostRecord(
                post_id=post_id,
                event_id=event_id,
                user_id=user_id,
                post_type=post_type,
                interaction_type="source" if post_type == "source" else "other",
                text_status="unknown" if post_type != "source" else classify_text_status(text_raw, missing_policy="unknown"),
                parent_post_id=parent_post_id,
                root_post_id=source_post_id,
                created_at=None,
                text_raw=text_raw,
                lang="en",
                engagement={"like_count": None, "reply_count": None, "repost_count": None},
                has_media=None,
                has_url=None,
                extra={
                    "subset": subset,
                    "tweet_id_raw": tweet_id_raw,
                    "uid_raw": representative_uid,
                    "uid_candidate_count": len(set(state["uid_candidates"])),
                    "time_delay_minutes_raw_values": sorted(state["raw_delay_values"]),
                    "time_delay_minutes_bucketed": delay_bucket,
                    "time_delay_hours_bucketed": round(delay_bucket / 60.0, 6),
                    "from_tree_only": not state["is_source"],
                    "tree_path": str(tree_path) if state["is_source"] else None,
                    "source_text_available": source_text_available,
                },
            )

        for post_id, count in reply_counts.items():
            if post_id in self.posts_by_id:
                self.posts_by_id[post_id].engagement["reply_count"] = count

        if source_post_id in self.posts_by_id:
            self.posts_by_id[source_post_id].engagement["reply_count"] = reply_counts.get(source_post_id, 0)

        label_norm = (raw_label or "").strip().lower()
        if label_norm in ACL2017_LABELS:
            if label_norm == "non-rumor":
                rumour_status = "non-rumour"
                veracity = "non-rumour"
            else:
                rumour_status = "rumour"
                veracity = label_norm

            self.annotations.append(
                AnnotationRecord(
                    annotation_id=f"{self.dataset_id}:ann:rumour_status:{event_id}",
                    target_type="event",
                    target_id=event_id,
                    task="rumour_status",
                    label=rumour_status,
                    label_space="rumour_status",
                    original_label=label_norm,
                    anchor_post_id=source_post_id,
                    annotator_kind="gold",
                    confidence=None,
                    extra={"subset": subset, "source_file": "label.txt", "raw_label": raw_label},
                )
            )
            self.annotations.append(
                AnnotationRecord(
                    annotation_id=f"{self.dataset_id}:ann:veracity:{event_id}",
                    target_type="event",
                    target_id=event_id,
                    task="veracity",
                    label=veracity,
                    label_space="veracity",
                    original_label=label_norm,
                    anchor_post_id=source_post_id,
                    annotator_kind="mapped" if label_norm == "non-rumor" else "gold",
                    confidence=None,
                    extra={"subset": subset, "source_file": "label.txt", "raw_label": raw_label},
                )
            )
        else:
            self.validation_warnings.append(f"Missing or unknown label for source tweet {source_tweet_id} in {subset}")

        propagation = "full_tree"
        if not edges:
            propagation = "flat"
        elif all(self._post_key(edge[0]["tweet_id"], self._bucket_delay(edge[0]["delay"])) == source_node_key for edge in edges):
            propagation = "star"

        self.events.append(
            EventRecord(
                event_id=event_id,
                dataset_id=self.dataset_id,
                platform="twitter",
                source_post_id=source_post_id,
                split="all",
                topic=subset,
                lang="en",
                start_time=None,
                end_time=None,
                view_status={
                    "semantic": "available" if source_text else "none",
                    "stance": "none",
                    "propagation": propagation,
                    "evidence": "none",
                    "source": "partial",
                },
                extra={
                    "subset": subset,
                    "tree_path": str(tree_path),
                    "source_tweet_id_raw": source_tweet_id,
                    "label_raw": raw_label,
                    "reply_text_available": False,
                    "timestamp_available": False,
                    "reconstructed_time_available": True,
                    "reconstructed_time_unit": "minutes_since_source",
                    "reconstructed_max_delay_minutes": max_delay_minutes,
                    "reconstructed_max_delay_hours": round(max_delay_minutes / 60.0, 6),
                    "delay_bucket_minutes": self.delay_bucket_minutes,
                },
            )
        )

    def _load_label_map(self, path: Path) -> dict[str, str]:
        label_map: dict[str, str] = {}
        for line in _read_text_lines(path):
            text = line.strip()
            if not text or ":" not in text:
                continue
            label, source_tweet_id = text.split(":", 1)
            label_map[source_tweet_id.strip()] = label.strip()
        return label_map

    def _load_source_map(self, path: Path) -> dict[str, str]:
        source_map: dict[str, str] = {}
        for line in _read_text_lines(path):
            text = line.rstrip("\n")
            if not text or "\t" not in text:
                continue
            source_tweet_id, source_text = text.split("\t", 1)
            source_map[source_tweet_id.strip()] = source_text.strip() or None
        return source_map

    def _parse_tree_edges(self, path: Path) -> tuple[list[tuple[dict[str, str], dict[str, str]]], dict[str, str] | None]:
        edges: list[tuple[dict[str, str], dict[str, str]]] = []
        root_node: dict[str, str] | None = None
        for line in _read_text_lines(path):
            text = line.strip()
            if not text:
                continue
            nodes = TREE_NODE_PATTERN.findall(text)
            if len(nodes) != 2:
                self.validation_warnings.append(f"Unparseable tree edge in {path}: {text}")
                continue
            parent = {"uid": nodes[0][0], "tweet_id": nodes[0][1], "delay": nodes[0][2]}
            child = {"uid": nodes[1][0], "tweet_id": nodes[1][1], "delay": nodes[1][2]}
            if parent["uid"] == "ROOT" and parent["tweet_id"] == "ROOT":
                if root_node is None:
                    root_node = child
                continue
            edges.append((parent, child))
        return edges, root_node

    def _register_node(self, post_state: dict[tuple[str, float], dict[str, Any]], node: dict[str, str], is_source: bool) -> None:
        key = self._post_key(node["tweet_id"], self._bucket_delay(node["delay"]))
        state = post_state.setdefault(
            key,
            {
                "tweet_id_raw": node["tweet_id"],
                "delay_bucket_minutes": key[1],
                "raw_delay_values": set(),
                "uid_candidates": [],
                "parent_candidates": Counter(),
                "is_source": False,
            },
        )
        state["raw_delay_values"].add(str(node["delay"]))
        if node["uid"] not in state["uid_candidates"]:
            state["uid_candidates"].append(node["uid"])
        if is_source:
            state["is_source"] = True

    @staticmethod
    def _choose_parent_key(parent_candidates: Counter, source_key: tuple[str, float]) -> tuple[str, float] | None:
        if not parent_candidates:
            return source_key
        ranked = sorted(parent_candidates.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
        return ranked[0][0]

    @classmethod
    def _post_key(cls, tweet_id: str, delay_bucket: float) -> tuple[str, float]:
        return str(tweet_id).strip(), float(delay_bucket)

    @classmethod
    def _bucket_delay(cls, raw_delay: str) -> float:
        try:
            value = float(raw_delay)
        except (TypeError, ValueError):
            value = 0.0
        bucket = round(value / cls.delay_bucket_minutes) * cls.delay_bucket_minutes
        return round(bucket, 6)

    def _upsert_user(self, raw_uid: str, subset: str) -> str | None:
        uid = str(raw_uid).strip()
        if not uid:
            return None
        user_id = self._canonical_user_id(uid)
        if user_id not in self.users_by_id:
            self.users_by_id[user_id] = UserRecord(
                user_id=user_id,
                platform="twitter",
                account_created_at=None,
                verified=None,
                followers_count=None,
                following_count=None,
                post_count=None,
                profile_description_raw=None,
                display_name_raw=None,
                extra={"subset": subset, "uid_raw": uid, "from_tree_only": True},
            )
        return user_id

    def _validate(self) -> dict[str, Any]:
        event_map = {event.event_id: event for event in self.events}
        posts_by_event: dict[str, list[PostRecord]] = defaultdict(list)
        for post in self.posts_by_id.values():
            posts_by_event[post.event_id].append(post)

        for event in self.events:
            source_posts = [post for post in posts_by_event[event.event_id] if post.post_type == "source"]
            if len(source_posts) != 1:
                self.validation_errors.append(
                    f"Event {event.event_id} must have exactly one source post, found {len(source_posts)}"
                )
                continue
            source_post = source_posts[0]
            if source_post.post_id != event.source_post_id:
                self.validation_errors.append(
                    f"Event {event.event_id} source_post_id mismatch: {event.source_post_id} vs {source_post.post_id}"
                )
            if source_post.parent_post_id is not None:
                self.validation_errors.append(f"Source post {source_post.post_id} has non-null parent_post_id")
            if source_post.root_post_id != source_post.post_id:
                self.validation_errors.append(f"Source post {source_post.post_id} root_post_id must equal post_id")
            root_ids = {post.root_post_id for post in posts_by_event[event.event_id]}
            if root_ids != {event.source_post_id}:
                self.validation_errors.append(
                    f"Event {event.event_id} posts do not share a single root_post_id: {sorted(root_ids)}"
                )

        for annotation in self.annotations:
            if annotation.target_type == "event" and annotation.target_id not in event_map:
                self.validation_errors.append(f"Missing event annotation target: {annotation.target_id}")
            if annotation.target_type == "post" and annotation.target_id not in self.posts_by_id:
                self.validation_errors.append(f"Missing post annotation target: {annotation.target_id}")
        for post in self.posts_by_id.values():
            validate_post_schema(post, self.validation_errors)

        split_counts: dict[str, int] = {"all": 0}
        topic_counts: dict[str, int] = defaultdict(int)
        view_coverage: dict[str, dict[str, int]] = {
            key: defaultdict(int) for key in ["semantic", "stance", "propagation", "evidence", "source"]
        }
        for event in self.events:
            split_counts[event.split] = split_counts.get(event.split, 0) + 1
            if event.topic:
                topic_counts[event.topic] += 1
            for key, value in event.view_status.items():
                view_coverage[key][value] += 1

        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "events": len(self.events),
                "posts": len(self.posts_by_id),
                "users": len(self.users_by_id),
                "references": len(self.references),
                "annotations": len(self.annotations),
            },
            "split_counts": split_counts,
            "platform_counts": {"twitter": len(self.events)},
            "topic_counts": dict(topic_counts),
            "annotation_coverage": {
                "events_with_rumour_status_labels": len(
                    {ann.target_id for ann in self.annotations if ann.task == "rumour_status"}
                ),
                "events_with_veracity_labels": len({ann.target_id for ann in self.annotations if ann.task == "veracity"}),
            },
            "view_coverage": {key: dict(value) for key, value in view_coverage.items()},
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skipped_items": self.skipped_items,
            "schema_valid": len(self.validation_errors) == 0,
        }

    def _write_outputs(self, meta: DatasetMeta, report: dict[str, Any]) -> None:
        write_json(self.output_root / "dataset_meta.json", meta.to_dict())
        write_jsonl(self.output_root / "events.jsonl", [row.to_dict() for row in self.events])
        write_jsonl(self.output_root / "posts.jsonl", [row.to_dict() for row in self.posts_by_id.values()])
        write_jsonl(self.output_root / "users.jsonl", [row.to_dict() for row in self.users_by_id.values()])
        write_jsonl(self.output_root / "references.jsonl", [row.to_dict() for row in self.references])
        write_jsonl(self.output_root / "annotations.jsonl", [row.to_dict() for row in self.annotations])
        write_json(self.output_root / "validation_report.json", report)

    @classmethod
    def _canonical_event_id(cls, subset: str, source_tweet_id: str) -> str:
        return f"{cls.dataset_id}:event:{subset}:{source_tweet_id}"

    @classmethod
    def _canonical_post_id(cls, subset: str, tweet_id: str, delay_bucket: float) -> str:
        return f"{cls.dataset_id}:post:{subset}:{tweet_id}:{delay_bucket:.6f}"

    @classmethod
    def _canonical_user_id(cls, uid: str) -> str:
        return f"{cls.dataset_id}:user:twitter:{uid}"


class RumorDetectionACL2017SubsetCanonicalConverter(RumorDetectionACL2017CanonicalConverter):
    def __init__(self, input_root: str | Path, output_root: str | Path, subset: str, dataset_id: str) -> None:
        super().__init__(input_root, output_root)
        self.subset = subset.strip().lower()
        self.dataset_id = dataset_id

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="twitter",
            task_family="rumour_detection_and_propagation",
            time_format="iso_8601_utc",
            label_spaces={
                "rumour_status": ["rumour", "non-rumour"],
                "veracity": ["true", "false", "unverified", "non-rumour"],
            },
            notes=(
                f"Canonical conversion of the ACL 2017 {self.subset} subset. "
                "Source tweet text is available, while reply text and full user "
                "metadata are not included in the public raw release. Tree nodes "
                "are preserved as observable structural posts with null text/"
                "timestamp where the raw release omits them."
            ),
        )

        self._convert_subsets()
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_subsets(self) -> None:
        subset_root = self.input_root
        if not (subset_root / "label.txt").exists():
            subset_root = self.input_root / self.subset
        if not subset_root.exists():
            self.validation_warnings.append(f"Missing expected subset root: {subset_root}")
            return
        label_map = self._load_label_map(subset_root / "label.txt")
        source_map = self._load_source_map(subset_root / "source_tweets.txt")
        tree_root = subset_root / "tree"
        for tree_path in sorted(path for path in tree_root.glob("*.txt") if path.is_file()):
            source_tweet_id = tree_path.stem
            label = label_map.get(source_tweet_id)
            source_text = source_map.get(source_tweet_id)
            self._convert_thread(self.subset, tree_path, source_tweet_id, source_text, label)


class Twitter15CanonicalConverter(RumorDetectionACL2017SubsetCanonicalConverter):
    dataset_id = "Twitter15"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root, output_root, subset="twitter15", dataset_id=self.dataset_id)


class Twitter16CanonicalConverter(RumorDetectionACL2017SubsetCanonicalConverter):
    dataset_id = "Twitter16"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root, output_root, subset="twitter16", dataset_id=self.dataset_id)


class RumdectTwitterCanonicalConverter:
    dataset_id = "RumdectTwitter"
    label_space = {"rumour_status": ["rumour", "non-rumour"]}

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.input_root = input_root
        self.output_root = Path(output_root)
        self.events: list[EventRecord] = []
        self.posts_by_id: dict[str, PostRecord] = {}
        self.users_by_id: dict[str, UserRecord] = {}
        self.references: list[ReferenceRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []
        self.skipped_items: list[dict[str, Any]] = []

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="twitter",
            task_family="rumour_detection",
            time_format="unavailable",
            label_spaces=self.label_space,
            notes=(
                "Canonical conversion of the RumDect Twitter subset. The raw "
                "release contains event ids, rumour labels, tweet id lists, "
                "and optional event claims, but does not release tweet text, "
                "timestamps, or user metadata."
            ),
        )

        self._convert_events()
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_events(self) -> None:
        events_path = self.input_root / "Twitter.txt"
        claims_path = self.input_root / "Twitter_event_claims.txt"
        if not events_path.exists():
            self.validation_warnings.append(f"Missing expected raw file: {events_path}")
            return
        claim_map = self._load_claim_map(claims_path)

        for line in _read_text_lines(events_path):
            text = line.strip()
            if not text:
                continue
            try:
                eid_part, label_part, tweet_ids_part = text.split("\t", 2)
            except ValueError:
                self.skipped_items.append(
                    {"type": "event", "path": str(events_path), "reason": f"unparseable line: {text[:120]}"}
                )
                continue

            event_raw_id = eid_part.replace("eid:", "", 1).strip()
            label_raw = label_part.replace("label:", "", 1).strip()
            tweet_ids = [item.strip() for item in tweet_ids_part.split() if item.strip()]
            if not event_raw_id or not tweet_ids:
                self.skipped_items.append(
                    {
                        "type": "event",
                        "path": str(events_path),
                        "reason": f"missing event id or tweet ids: {text[:120]}",
                    }
                )
                continue

            event_id = self._canonical_event_id(event_raw_id)
            source_tweet_id = tweet_ids[0]
            source_post_id = self._canonical_post_id(event_raw_id, source_tweet_id)
            claim = claim_map.get(event_raw_id)

            self.events.append(
                EventRecord(
                    event_id=event_id,
                    dataset_id=self.dataset_id,
                    platform="twitter",
                    source_post_id=source_post_id,
                    split="all",
                    topic=None,
                    lang="en",
                    start_time=None,
                    end_time=None,
                    view_status={
                        "semantic": "claim_only" if claim else "ids_only",
                        "stance": "none",
                        "propagation": "flat_list",
                        "evidence": "none",
                        "source": "none",
                    },
                    extra={
                        "event_id_raw": event_raw_id,
                        "label_raw": label_raw,
                        "tweet_id_count": len(tweet_ids),
                        "claim_available": claim is not None,
                    },
                )
            )

            self.posts_by_id[source_post_id] = PostRecord(
                post_id=source_post_id,
                event_id=event_id,
                user_id=None,
                post_type="source",
                interaction_type="source",
                text_status=classify_text_status(claim, missing_policy="unknown"),
                parent_post_id=None,
                root_post_id=source_post_id,
                created_at=None,
                text_raw=claim,
                lang="en",
                engagement={"like_count": None, "reply_count": max(len(tweet_ids) - 1, 0), "repost_count": None},
                has_media=None,
                has_url=bool(claim and URL_PATTERN.search(claim)),
                extra={
                    "event_id_raw": event_raw_id,
                    "tweet_id_raw": source_tweet_id,
                    "claim_surrogate_text": claim is not None,
                    "content_source": "Twitter_event_claims.txt" if claim else None,
                },
            )

            for reply_tweet_id in tweet_ids[1:]:
                post_id = self._canonical_post_id(event_raw_id, reply_tweet_id)
                self.posts_by_id[post_id] = PostRecord(
                    post_id=post_id,
                    event_id=event_id,
                    user_id=None,
                    post_type="response",
                    interaction_type="other",
                    text_status="unknown",
                    parent_post_id=source_post_id,
                    root_post_id=source_post_id,
                    created_at=None,
                    text_raw=None,
                    lang="en",
                    engagement={"like_count": None, "reply_count": None, "repost_count": None},
                    has_media=None,
                    has_url=None,
                    extra={
                        "event_id_raw": event_raw_id,
                        "tweet_id_raw": reply_tweet_id,
                        "content_available": False,
                        "metadata_available": False,
                    },
                )

            label = self._map_label(label_raw)
            if label is None:
                self.validation_warnings.append(f"Unknown label {label_raw!r} for event {event_raw_id}")
            else:
                self.annotations.append(
                    AnnotationRecord(
                        annotation_id=f"{self.dataset_id}:ann:rumour_status:{event_id}",
                        target_type="event",
                        target_id=event_id,
                        task="rumour_status",
                        label=label,
                        label_space="rumour_status",
                        original_label=label_raw,
                        anchor_post_id=source_post_id,
                        annotator_kind="gold",
                        confidence=None,
                        extra={"source_file": "Twitter.txt", "event_id_raw": event_raw_id},
                    )
                )

    @staticmethod
    def _load_claim_map(path: Path) -> dict[str, str]:
        claim_map: dict[str, str] = {}
        if not path.exists():
            return claim_map
        for line in _read_text_lines(path):
            text = line.rstrip("\n")
            if not text or "\t" not in text:
                continue
            event_part, claim = text.split("\t", 1)
            event_raw_id = event_part.replace("eid:", "", 1).strip()
            claim_map[event_raw_id] = claim.strip() or None
        return claim_map

    @staticmethod
    def _map_label(label_raw: str) -> str | None:
        normalized = str(label_raw).strip()
        if normalized == "1":
            return "rumour"
        if normalized == "0":
            return "non-rumour"
        return None

    def _validate(self) -> dict[str, Any]:
        event_map = {event.event_id: event for event in self.events}
        posts_by_event: dict[str, list[PostRecord]] = defaultdict(list)
        for post in self.posts_by_id.values():
            posts_by_event[post.event_id].append(post)

        for event in self.events:
            source_posts = [post for post in posts_by_event[event.event_id] if post.post_type == "source"]
            if len(source_posts) != 1:
                self.validation_errors.append(
                    f"Event {event.event_id} must have exactly one source post, found {len(source_posts)}"
                )
                continue
            source_post = source_posts[0]
            if source_post.post_id != event.source_post_id:
                self.validation_errors.append(
                    f"Event {event.event_id} source_post_id mismatch: {event.source_post_id} vs {source_post.post_id}"
                )
            if source_post.parent_post_id is not None:
                self.validation_errors.append(f"Source post {source_post.post_id} has non-null parent_post_id")

        for annotation in self.annotations:
            if annotation.target_type == "event" and annotation.target_id not in event_map:
                self.validation_errors.append(f"Missing event annotation target: {annotation.target_id}")
        for post in self.posts_by_id.values():
            validate_post_schema(post, self.validation_errors)

        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "events": len(self.events),
                "posts": len(self.posts_by_id),
                "users": len(self.users_by_id),
                "references": len(self.references),
                "annotations": len(self.annotations),
            },
            "split_counts": {"all": len(self.events)},
            "platform_counts": {"twitter": len(self.events)},
            "annotation_coverage": {
                "events_with_rumour_status_labels": len(
                    {ann.target_id for ann in self.annotations if ann.task == "rumour_status"}
                )
            },
            "view_coverage": {
                "semantic": dict(Counter(event.view_status["semantic"] for event in self.events)),
                "stance": dict(Counter(event.view_status["stance"] for event in self.events)),
                "propagation": dict(Counter(event.view_status["propagation"] for event in self.events)),
                "evidence": dict(Counter(event.view_status["evidence"] for event in self.events)),
                "source": dict(Counter(event.view_status["source"] for event in self.events)),
            },
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skipped_items": self.skipped_items,
            "schema_valid": len(self.validation_errors) == 0,
        }

    def _write_outputs(self, meta: DatasetMeta, report: dict[str, Any]) -> None:
        write_json(self.output_root / "dataset_meta.json", meta.to_dict())
        write_jsonl(self.output_root / "events.jsonl", [row.to_dict() for row in self.events])
        write_jsonl(self.output_root / "posts.jsonl", [row.to_dict() for row in self.posts_by_id.values()])
        write_jsonl(self.output_root / "users.jsonl", [row.to_dict() for row in self.users_by_id.values()])
        write_jsonl(self.output_root / "references.jsonl", [row.to_dict() for row in self.references])
        write_jsonl(self.output_root / "annotations.jsonl", [row.to_dict() for row in self.annotations])
        write_json(self.output_root / "validation_report.json", report)

    @classmethod
    def _canonical_event_id(cls, event_raw_id: str) -> str:
        return f"{cls.dataset_id}:event:twitter:{event_raw_id}"

    @classmethod
    def _canonical_post_id(cls, event_raw_id: str, tweet_id: str) -> str:
        return f"{cls.dataset_id}:post:twitter:{event_raw_id}:{tweet_id}"


class RumdectWeiboCanonicalConverter:
    dataset_id = "Weibo"
    label_space = {"rumour_status": ["rumour", "non-rumour"]}

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.input_root = input_root
        self.output_root = Path(output_root)
        self.events: list[EventRecord] = []
        self.posts_by_id: dict[str, PostRecord] = {}
        self.users_by_id: dict[str, UserRecord] = {}
        self.references: list[ReferenceRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []
        self.skipped_items: list[dict[str, Any]] = []

    def convert(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        meta = DatasetMeta(
            dataset_id=self.dataset_id,
            platform="weibo",
            task_family="rumour_detection_and_propagation",
            time_format="unix_to_iso_8601_utc",
            label_spaces=self.label_space,
            notes=(
                "Canonical conversion of the Weibo subset. The raw "
                "release provides event labels, ordered post id lists, and "
                "per-event JSON files containing post text, user metadata, "
                "parent links, and timestamps."
            ),
        )

        self._convert_events()
        CANONICAL_LOGGER.info(
            "weibo canonical convert finished extraction: events=%s posts=%s users=%s refs=%s anns=%s elapsed=%.2fs",
            len(self.events),
            len(self.posts_by_id),
            len(self.users_by_id),
            len(self.references),
            len(self.annotations),
            time.perf_counter() - started_at,
        )
        refresh_stance_view_status(self.events, self.posts_by_id, self.annotations)
        report = self._validate()
        self._write_outputs(meta, report)
        return report

    def _convert_events(self) -> None:
        events_index = self._load_weibo_index(self.input_root / "Weibo.txt")
        tasks = [
            (
                index,
                event_raw_id,
                payload["label_raw"],
                payload["post_ids"],
                _worker_path_string(self.input_root / "Weibo" / f"{event_raw_id}.json"),
            )
            for index, (event_raw_id, payload) in enumerate(events_index.items())
        ]
        total_events = len(tasks)
        workers = self._weibo_worker_count(total_events)
        started_at = time.perf_counter()
        CANONICAL_LOGGER.info(
            "weibo canonical convert starting: events=%s workers=%s input_root=%s",
            total_events,
            workers,
            self.input_root,
        )

        if workers <= 1 or total_events <= 1:
            for completed, task in enumerate(tasks, start=1):
                self._merge_weibo_result(_convert_single_weibo_event(task))
                if completed % WEIBO_PROGRESS_EVERY == 0 or completed == total_events:
                    CANONICAL_LOGGER.info(
                        "weibo canonical convert progress: %s/%s events elapsed=%.2fs",
                        completed,
                        total_events,
                        time.perf_counter() - started_at,
                    )
            return

        results: list[dict[str, Any] | None] = [None] * total_events
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_convert_single_weibo_event, task): task[0] for task in tasks}
            completed = 0
            for future in as_completed(future_map):
                result = future.result()
                results[result["index"]] = result
                completed += 1
                if completed % WEIBO_PROGRESS_EVERY == 0 or completed == total_events:
                    CANONICAL_LOGGER.info(
                        "weibo canonical convert progress: %s/%s events elapsed=%.2fs",
                        completed,
                        total_events,
                        time.perf_counter() - started_at,
                    )
        for result in results:
            if result is not None:
                self._merge_weibo_result(result)
        if not self.events and self.skipped_items:
            for sample in self.skipped_items[:3]:
                CANONICAL_LOGGER.warning("weibo canonical skipped sample: %s", sample)

    @staticmethod
    def _load_weibo_index(path: Path) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for line in _read_text_lines(path):
            text = line.strip()
            if not text:
                continue
            try:
                event_part, label_part, post_ids_part = text.split("\t", 2)
            except ValueError:
                continue
            event_raw_id = event_part.replace("eid:", "", 1).strip()
            label_raw = label_part.replace("label:", "", 1).strip()
            post_ids = [item.strip() for item in post_ids_part.split() if item.strip()]
            index[event_raw_id] = {"label_raw": label_raw, "post_ids": post_ids}
        return index

    def _convert_weibo_event(
        self,
        event_raw_id: str,
        label_raw: str,
        indexed_post_ids: list[str],
        rows: list[dict[str, Any]],
        json_path: Path,
    ) -> None:
        event_id = self._canonical_event_id(event_raw_id)
        posts_by_id_raw = {}
        posts_by_mid_raw = {}
        created_times: list[str] = []
        source_row = None

        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_id = str(row.get("id") or "").strip()
            raw_mid = str(row.get("mid") or "").strip()
            if raw_id:
                posts_by_id_raw[raw_id] = row
            if raw_mid:
                posts_by_mid_raw[raw_mid] = row
            parent_raw = row.get("parent")
            if source_row is None and parent_raw in (None, "", "None", 0, "0"):
                source_row = row

        if source_row is None:
            source_row = rows[0] if rows and isinstance(rows[0], dict) else None
        if source_row is None:
            self.skipped_items.append({"type": "event", "path": str(json_path), "reason": "missing source row"})
            return

        source_raw_id = str(source_row.get("id") or event_raw_id).strip()
        source_post_id = self._canonical_post_id(event_raw_id, source_raw_id)

        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_id = str(row.get("id") or "").strip()
            if not raw_id:
                self.skipped_items.append({"type": "post", "path": str(json_path), "reason": "missing post id"})
                continue
            post_id = self._canonical_post_id(event_raw_id, raw_id)
            raw_parent = row.get("parent")
            parent_post_id = None
            post_type = "response"
            if raw_parent in (None, "", "None", 0, "0") or raw_id == source_raw_id:
                post_type = "source" if raw_id == source_raw_id else "response"
                parent_post_id = None if raw_id == source_raw_id else source_post_id
            else:
                parent_key = str(raw_parent).strip()
                parent_row = posts_by_mid_raw.get(parent_key) or posts_by_id_raw.get(parent_key)
                if parent_row is not None:
                    parent_post_id = self._canonical_post_id(event_raw_id, str(parent_row.get("id")).strip())
                else:
                    parent_post_id = source_post_id

            created_at, raw_time = _parse_unix_time(row.get("t"))
            if created_at:
                created_times.append(created_at)
            text_raw = None
            if isinstance(row.get("text"), str) and row["text"].strip():
                text_raw = row["text"].strip()
            elif isinstance(row.get("original_text"), str) and row["original_text"].strip():
                text_raw = row["original_text"].strip()

            text_status = classify_text_status(
                text_raw,
                missing_policy="unknown" if raw_id == source_raw_id else "empty",
                boilerplate_texts=WEIBO_BOILERPLATE_NORMALIZED,
            )
            interaction_type = "source"
            if raw_id != source_raw_id:
                interaction_type = "quote_repost" if text_status == "substantive" else "repost"

            user_id = self._upsert_weibo_user(row)
            refs = self._extract_weibo_refs(
                event_id=event_id,
                post_id=post_id,
                row=row,
                text_raw=text_raw,
                parent_post_id=parent_post_id if raw_id != source_raw_id else None,
                interaction_type=interaction_type,
            )
            self.references.extend(refs)

            self.posts_by_id[post_id] = PostRecord(
                post_id=post_id,
                event_id=event_id,
                user_id=user_id,
                post_type="source" if raw_id == source_raw_id else "response",
                interaction_type=interaction_type,
                text_status=text_status,
                parent_post_id=parent_post_id,
                root_post_id=source_post_id,
                created_at=created_at,
                text_raw=text_raw,
                lang="zh",
                engagement={
                    "like_count": self._to_int(row.get("attitudes_count")),
                    "reply_count": self._to_int(row.get("comments_count")),
                    "repost_count": self._to_int(row.get("reposts_count")),
                },
                has_media=bool(row.get("picture")),
                has_url=bool(text_raw and URL_PATTERN.search(text_raw)),
                extra={
                    "event_id_raw": event_raw_id,
                    "post_id_raw": raw_id,
                    "mid_raw": str(row.get("mid") or "").strip() or None,
                    "parent_raw": raw_parent,
                    "raw_time_unix": raw_time if raw_time is not None else row.get("t"),
                    "geo_raw": row.get("geo"),
                    "province": row.get("province"),
                    "city": row.get("city"),
                    "indexed_post_ids_contains_post": raw_id in indexed_post_ids,
                },
            )

        propagation = "full_tree"
        if len(self.posts_by_id) == 1:
            propagation = "flat"
        elif all(
            post.parent_post_id in (None, source_post_id) or post.post_id == source_post_id
            for post in self.posts_by_id.values()
            if post.event_id == event_id
        ):
            propagation = "star"

        response_text_available = any(
            post.event_id == event_id and post.post_type == "response" and post.text_status == "substantive"
            for post in self.posts_by_id.values()
        )

        self.events.append(
            EventRecord(
                event_id=event_id,
                dataset_id=self.dataset_id,
                platform="weibo",
                source_post_id=source_post_id,
                split="all",
                topic=None,
                lang="zh",
                start_time=min(created_times) if created_times else None,
                end_time=max(created_times) if created_times else None,
                view_status={
                    "semantic": "available",
                    "stance": "text_only" if response_text_available else "none",
                    "propagation": propagation,
                    "evidence": "none",
                    "source": "partial",
                },
                extra={
                    "event_id_raw": event_raw_id,
                    "json_path": str(json_path),
                    "indexed_post_count": len(indexed_post_ids),
                    "json_post_count": len([row for row in rows if isinstance(row, dict)]),
                    "label_raw": label_raw,
                },
            )
        )

        label = RumdectTwitterCanonicalConverter._map_label(label_raw)
        if label is None:
            self.validation_warnings.append(f"Unknown label {label_raw!r} for Weibo event {event_raw_id}")
        else:
            self.annotations.append(
                AnnotationRecord(
                    annotation_id=f"{self.dataset_id}:ann:rumour_status:{event_id}",
                    target_type="event",
                    target_id=event_id,
                    task="rumour_status",
                    label=label,
                    label_space="rumour_status",
                    original_label=label_raw,
                    anchor_post_id=source_post_id,
                    annotator_kind="gold",
                    confidence=None,
                    extra={"source_file": "Weibo.txt", "event_id_raw": event_raw_id},
                )
            )

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _upsert_weibo_user(self, row: dict[str, Any]) -> str | None:
        uid = str(row.get("uid") or "").strip()
        if not uid:
            return None
        user_id = self._canonical_user_id(uid)
        if user_id not in self.users_by_id:
            account_created_at, raw_created = _parse_unix_time(row.get("user_created_at"))
            self.users_by_id[user_id] = UserRecord(
                user_id=user_id,
                platform="weibo",
                account_created_at=account_created_at,
                verified=bool(row.get("verified")) if row.get("verified") is not None else None,
                followers_count=self._to_int(row.get("followers_count")),
                following_count=self._to_int(row.get("friends_count")),
                post_count=self._to_int(row.get("statuses_count")),
                profile_description_raw=row.get("user_description") if isinstance(row.get("user_description"), str) else None,
                display_name_raw=row.get("screen_name") if isinstance(row.get("screen_name"), str) else None,
                extra={
                    "uid_raw": uid,
                    "username": row.get("username"),
                    "gender": row.get("gender"),
                    "verified_reason": row.get("verified_reason"),
                    "verified_type": row.get("verified_type"),
                    "bi_followers_count": self._to_int(row.get("bi_followers_count")),
                    "favourites_count": self._to_int(row.get("favourites_count")),
                    "user_location": row.get("user_location"),
                    "user_avatar": row.get("user_avatar"),
                    "user_geo_enabled": row.get("user_geo_enabled"),
                    "user_created_at_raw": raw_created if raw_created is not None else row.get("user_created_at"),
                },
            )
        return user_id

    def _extract_weibo_refs(
        self,
        event_id: str,
        post_id: str,
        row: dict[str, Any],
        text_raw: str | None,
        parent_post_id: str | None,
        interaction_type: str,
    ) -> list[ReferenceRecord]:
        refs: list[ReferenceRecord] = []
        ref_index = 0
        seen: set[tuple[str, str]] = set()

        for field_name in ["text", "original_text"]:
            value = row.get(field_name)
            if isinstance(value, str):
                for match in URL_PATTERN.findall(value):
                    normalized = normalize_url(match)
                    key = ("url", normalized)
                    if key in seen:
                        continue
                    seen.add(key)
                    refs.append(
                        ReferenceRecord(
                            ref_id=f"{self.dataset_id}:ref:{post_id}:{ref_index}",
                            event_id=event_id,
                            post_id=post_id,
                            ref_type="url",
                            raw_value=match,
                            normalized_value=normalized,
                            domain=urlparse(normalized).netloc.lower() or None,
                            target_post_id=None,
                            extra={"source": field_name},
                        )
                    )
                    ref_index += 1

        picture = row.get("picture")
        if isinstance(picture, str) and picture.strip():
            normalized = normalize_url(picture)
            refs.append(
                ReferenceRecord(
                    ref_id=f"{self.dataset_id}:ref:{post_id}:{ref_index}",
                    event_id=event_id,
                    post_id=post_id,
                    ref_type="media",
                    raw_value=picture,
                    normalized_value=normalized,
                    domain=urlparse(normalized).netloc.lower() or None,
                    target_post_id=None,
                    extra={"source": "picture"},
                )
            )
            ref_index += 1

        if parent_post_id and interaction_type in {"repost", "quote_repost"}:
            refs.append(
                ReferenceRecord(
                    ref_id=f"{self.dataset_id}:ref:{post_id}:repost",
                    event_id=event_id,
                    post_id=post_id,
                    ref_type="reposted_post",
                    raw_value=None,
                    normalized_value=None,
                    domain=None,
                    target_post_id=parent_post_id,
                    extra={"source": "parent_post_id"},
                )
            )

        return refs

    def _validate(self) -> dict[str, Any]:
        event_map = {event.event_id: event for event in self.events}
        posts_by_event: dict[str, list[PostRecord]] = defaultdict(list)
        for post in self.posts_by_id.values():
            posts_by_event[post.event_id].append(post)

        for event in self.events:
            event_posts = posts_by_event[event.event_id]
            source_posts = [post for post in event_posts if post.post_type == "source"]
            if len(source_posts) != 1:
                self.validation_errors.append(
                    f"Event {event.event_id} must have exactly one source post, found {len(source_posts)}"
                )
                continue
            source_post = source_posts[0]
            if source_post.post_id != event.source_post_id:
                self.validation_errors.append(
                    f"Event {event.event_id} source_post_id mismatch: {event.source_post_id} vs {source_post.post_id}"
                )
            if source_post.parent_post_id is not None:
                self.validation_errors.append(f"Source post {source_post.post_id} has non-null parent_post_id")
            root_ids = {post.root_post_id for post in event_posts}
            if root_ids != {event.source_post_id}:
                self.validation_errors.append(
                    f"Event {event.event_id} posts do not share a single root_post_id: {sorted(root_ids)}"
                )

        for annotation in self.annotations:
            if annotation.target_type == "event" and annotation.target_id not in event_map:
                self.validation_errors.append(f"Missing event annotation target: {annotation.target_id}")
        for post in self.posts_by_id.values():
            validate_post_schema(post, self.validation_errors)

        return {
            "dataset_id": self.dataset_id,
            "counts": {
                "events": len(self.events),
                "posts": len(self.posts_by_id),
                "users": len(self.users_by_id),
                "references": len(self.references),
                "annotations": len(self.annotations),
            },
            "split_counts": {"all": len(self.events)},
            "platform_counts": {"weibo": len(self.events)},
            "annotation_coverage": {
                "events_with_rumour_status_labels": len(
                    {ann.target_id for ann in self.annotations if ann.task == "rumour_status"}
                )
            },
            "view_coverage": {
                "semantic": dict(Counter(event.view_status["semantic"] for event in self.events)),
                "stance": dict(Counter(event.view_status["stance"] for event in self.events)),
                "propagation": dict(Counter(event.view_status["propagation"] for event in self.events)),
                "evidence": dict(Counter(event.view_status["evidence"] for event in self.events)),
                "source": dict(Counter(event.view_status["source"] for event in self.events)),
            },
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skipped_items": self.skipped_items,
            "schema_valid": len(self.validation_errors) == 0,
        }

    def _write_outputs(self, meta: DatasetMeta, report: dict[str, Any]) -> None:
        write_json(self.output_root / "dataset_meta.json", meta.to_dict())
        write_jsonl(self.output_root / "events.jsonl", [row.to_dict() for row in self.events])
        write_jsonl(self.output_root / "posts.jsonl", [row.to_dict() for row in self.posts_by_id.values()])
        write_jsonl(self.output_root / "users.jsonl", [row.to_dict() for row in self.users_by_id.values()])
        write_jsonl(self.output_root / "references.jsonl", [row.to_dict() for row in self.references])
        write_jsonl(self.output_root / "annotations.jsonl", [row.to_dict() for row in self.annotations])
        write_json(self.output_root / "validation_report.json", report)

    @classmethod
    def _canonical_event_id(cls, event_raw_id: str) -> str:
        return f"{cls.dataset_id}:event:weibo:{event_raw_id}"

    @classmethod
    def _canonical_post_id(cls, event_raw_id: str, post_raw_id: str) -> str:
        if event_raw_id == post_raw_id:
            # Keep the source-post id compact. The event already scopes the root.
            return f"{cls.dataset_id}:post:weibo:{event_raw_id}"
        return f"{cls.dataset_id}:post:weibo:{event_raw_id}:{post_raw_id}"

    @classmethod
    def _canonical_user_id(cls, uid: str) -> str:
        return f"{cls.dataset_id}:user:weibo:{uid}"

    @staticmethod
    def _weibo_worker_count(total_events: int) -> int:
        raw_value = os.environ.get("UMCM_WEIBO_CONVERTER_WORKERS", "").strip()
        if raw_value:
            try:
                return max(1, min(total_events, int(raw_value)))
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 1
        return max(1, min(total_events, min(8, cpu_count)))

    def _merge_weibo_result(self, result: dict[str, Any]) -> None:
        event = result.get("event")
        if event is not None:
            self.events.append(event)
        for post in result.get("posts", []):
            self.posts_by_id[post.post_id] = post
        for user in result.get("users", []):
            self.users_by_id[user.user_id] = user
        self.references.extend(result.get("references", []))
        annotation = result.get("annotation")
        if annotation is not None:
            self.annotations.append(annotation)
        self.validation_warnings.extend(result.get("validation_warnings", []))
        self.skipped_items.extend(result.get("skipped_items", []))

