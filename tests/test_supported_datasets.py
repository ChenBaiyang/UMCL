from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from config.resolver import resolve_encoder_config, resolve_event_pipeline_config, resolve_fusion_config
from config.types import DATASET_IDS
from preprocess.canonical.converters import PhemeVeracityCanonicalConverter, _map_weibo_label
from preprocess.event_pipeline.cli_build import _build_split_assignments
from preprocess.event_pipeline.dataset_ingest import available_dataset_ids
from utils.primary_label_space import resolve_primary_task_spec


DATASETS = ["PHEME", "Twitter15", "Twitter16", "Weibo"]


@pytest.mark.parametrize("raw, expected", [("1", "rumour"), (1, "rumour"),
                                         (" 0 ", "non-rumour"), (None, None), ("other", None)])
def test_weibo_label_mapping_keeps_previous_semantics(raw, expected):
    assert _map_weibo_label(raw) == expected


@pytest.mark.parametrize("dataset", DATASETS)
def test_supported_dataset_contract_and_stratified_splits(dataset):
    assert DATASET_IDS == available_dataset_ids() == DATASETS
    assert resolve_event_pipeline_config(dataset).dataset_id == dataset
    assert resolve_fusion_config(dataset).dataset_id == dataset
    for view in ["content", "stance", "propagation", "evidence", "source"]:
        assert resolve_encoder_config(view, dataset).dataset_id == dataset
    spec = resolve_primary_task_spec(dataset)
    assert len(spec["class_names"]) == (2 if dataset == "Weibo" else 4)
    events = [
        {"event_id": f"event-{i}", "task_targets": {
            spec["task_name"]: {"label": spec["class_names"][i % len(spec["class_names"])]}
        }}
        for i in range(20)
    ]
    assignments = _build_split_assignments(dataset, events, [], 2)
    assert assignments == _build_split_assignments(dataset, events, [], 2)
    assert {key: sum(a.canonical_split == key for a in assignments.values())
            for key in ["train", "val", "test"]} == {"train": 14, "val": 3, "test": 3}


def test_unknown_dataset_is_rejected_before_artifact_build():
    for resolver in [resolve_event_pipeline_config, resolve_fusion_config, resolve_primary_task_spec]:
        with pytest.raises(ValueError):
            resolver("retired-dataset")
    with pytest.raises(ValueError):
        resolve_encoder_config("content", "retired-dataset")


def test_pheme_thread_keeps_nested_parent_and_profile_parsing():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        thread = root / "raw" / "all-rnr-annotated-threads" / "topic" / "rumours" / "100"
        (thread / "source-tweets").mkdir(parents=True)
        (thread / "reactions").mkdir()
        user = {"id_str": "10", "verified": "true", "followers_count": "42",
                "friends_count": "7", "statuses_count": "9", "description": " profile ",
                "name": " author ", "created_at": "Wed Jan 01 00:00:00 +0000 2014"}
        source = {"id_str": "100", "text": "A source claim", "lang": " en ", "user": user,
                  "favorite_count": "3", "retweet_count": "4",
                  "created_at": "Wed Jan 07 11:00:00 +0000 2015"}
        reply = {"id_str": "101", "text": "A reply", "user": {"id_str": "11"},
                 "in_reply_to_status_id_str": "t3_100",
                 "created_at": "Wed Jan 07 11:01:00 +0000 2015"}
        nested = {"id_str": "102", "text": "Nested reply", "user": {"id_str": "12"},
                  "created_at": "Wed Jan 07 11:02:00 +0000 2015"}
        payloads = {"source-tweets/100.json": source, "reactions/101.json": reply,
                    "reactions/102.json": nested, "structure.json": {"100": {"101": {"102": []}}},
                    "annotation.json": {"is_rumour": "rumour", "misinformation": 0, "true": 1}}
        for name, value in payloads.items():
            (thread / name).write_text(json.dumps(value), encoding="utf-8")
        converter = PhemeVeracityCanonicalConverter(root / "raw", root / "canonical")
        report = converter.convert()
        assert report["schema_valid"]
        posts = converter.posts_by_id
        assert posts["PHEME:post:twitter:101"].parent_post_id == "PHEME:post:twitter:100"
        assert posts["PHEME:post:twitter:102"].parent_post_id == "PHEME:post:twitter:101"
        assert posts["PHEME:post:twitter:102"].extra["parent_inferred"] is True
        assert posts["PHEME:post:twitter:100"].engagement["like_count"] == 3
        profile = converter.users_by_id["PHEME:user:twitter:10"]
        assert profile.verified is True
        assert profile.followers_count == 42
        assert profile.profile_description_raw == "profile"
        assert converter.events[0].lang == "en"
        assert {a.task: a.label for a in converter.annotations} == {
            "rumour_status": "rumour", "veracity": "true"}
