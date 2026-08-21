from __future__ import annotations

from typing import Any

import numpy as np

from models.encoders.evidence_view.authority_bucketing import bucket_authority, normalize_source_channel, source_channel_id
from models.encoders.evidence_view.config import EvidenceViewConfig
from models.encoders.evidence_view.schemas import EvidenceReferenceRowMeta


def build_reference_rows(
    config: EvidenceViewConfig,
    dataset_id: str,
    platform: str,
    event_id: str,
    refs: list[dict[str, Any]],
) -> tuple[list[EvidenceReferenceRowMeta], np.ndarray]:
    meta_rows: list[EvidenceReferenceRowMeta] = []
    feature_rows: list[list[float]] = []
    for ref_order, ref in enumerate(refs):
        ref_type = ref.get("ref_type") or "unknown"
        authority_bucket_id, authority_bucket_name = bucket_authority(
            config=config,
            domain=ref.get("domain"),
            is_internal_platform_ref=ref.get("is_internal_platform_ref"),
            ref_type=ref_type,
            normalized_value=ref.get("normalized_value"),
        )
        source_name = normalize_source_channel(config, ((ref.get("extra") or {}).get("source")))
        meta_rows.append(
            EvidenceReferenceRowMeta(
                event_id=event_id,
                dataset_id=dataset_id,
                platform=platform,
                ref_id=ref.get("ref_id") or "",
                ref_type=ref_type,
                domain=ref.get("domain"),
                authority_bucket_name=authority_bucket_name,
                is_internal_platform_ref=bool(ref.get("is_internal_platform_ref")) if ref.get("is_internal_platform_ref") is not None else None,
                has_normalized_value=bool(ref.get("normalized_value")),
                has_domain=bool(ref.get("domain")),
                has_target_post_id=bool(ref.get("target_post_id")),
                source_channel=source_name,
                ref_order=ref_order,
                notes="",
            )
        )
        feature_rows.append(
            [
                float(config.ref_type_vocab.get(ref_type, config.ref_type_vocab["unknown"])),
                float(authority_bucket_id),
                float(1 if ref.get("normalized_value") else 0),
                float(1 if ref.get("domain") else 0),
                float(1 if ref.get("target_post_id") else 0),
                float(1 if ref.get("is_internal_platform_ref") is True else 0),
                float(1 if ref.get("is_internal_platform_ref") is False else 0),
                float(source_channel_id(config, (ref.get("extra") or {}).get("source"))),
            ]
        )
    if not feature_rows:
        return meta_rows, np.zeros((0, 8), dtype=np.float32)
    return meta_rows, np.asarray(feature_rows, dtype=np.float32)
