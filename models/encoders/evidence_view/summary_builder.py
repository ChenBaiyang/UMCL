from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from models.encoders.evidence_view.authority_bucketing import bucket_authority
from models.encoders.evidence_view.config import EvidenceViewConfig
from models.encoders.evidence_view.quality_masks import authority_unknown_mask, evidence_signal_observed, no_reference_rows_mask, sparse_evidence_mask, weak_evidence_mask


def build_summary_features(config: EvidenceViewConfig, event_row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    content = event_row.get("content") or {}
    surface = content.get("surface") or {}
    evidence = event_row.get("evidence") or {}
    refs = evidence.get("references") or []

    has_source_url_flag = bool(surface.get("has_url"))
    has_source_media_flag = bool(surface.get("has_media"))
    has_explicit_ref = bool(refs)
    has_any_signal = bool(has_explicit_ref or has_source_url_flag or has_source_media_flag)

    ref_types = Counter((ref.get("ref_type") or "unknown") for ref in refs)
    n_refs_total = len(refs)
    n_url_refs = int(ref_types.get("url", 0))
    n_media_refs = int(ref_types.get("media", 0))
    n_reference_types_present = sum(1 for count in ref_types.values() if count > 0)
    has_url_ref_type = 1 if n_url_refs > 0 else 0
    has_media_ref_type = 1 if n_media_refs > 0 else 0
    has_multi_ref_type = 1 if n_reference_types_present > 1 else 0
    has_multi_ref_rows = 1 if n_refs_total > 1 else 0

    n_refs_with_normalized_value = sum(1 for ref in refs if ref.get("normalized_value"))
    n_refs_with_domain = sum(1 for ref in refs if ref.get("domain"))
    n_target_post_id_refs = sum(1 for ref in refs if ref.get("target_post_id"))
    has_normalized_reference = 1 if n_refs_with_normalized_value > 0 else 0
    has_domain = 1 if n_refs_with_domain > 0 else 0
    has_target_post_id = 1 if n_target_post_id_refs > 0 else 0
    has_specific_reference_signal = 1 if (has_normalized_reference or has_domain or has_target_post_id) else 0

    n_internal_refs = sum(1 for ref in refs if ref.get("is_internal_platform_ref") is True)
    n_external_refs = sum(1 for ref in refs if ref.get("is_internal_platform_ref") is False)
    has_internal_ref = 1 if n_internal_refs > 0 else 0
    has_external_ref = 1 if n_external_refs > 0 else 0

    authority_counts = Counter()
    meaningful_authority_count = 0
    for ref in refs:
        _, bucket_name = bucket_authority(
            config=config,
            domain=ref.get("domain"),
            is_internal_platform_ref=ref.get("is_internal_platform_ref"),
            ref_type=ref.get("ref_type"),
            normalized_value=ref.get("normalized_value"),
        )
        authority_counts[bucket_name] += 1
        if bucket_name != "unknown":
            meaningful_authority_count += 1

    quality = {
        "evidence_signal_observed": evidence_signal_observed(has_explicit_ref, has_source_url_flag, has_source_media_flag),
        "no_reference_rows_mask": no_reference_rows_mask(n_refs_total),
        "weak_evidence_mask": weak_evidence_mask(has_any_signal, n_refs_total),
        "sparse_evidence_mask": sparse_evidence_mask(n_refs_total, bool(has_multi_ref_type), bool(has_multi_ref_rows)),
        "authority_unknown_mask": authority_unknown_mask(n_refs_total, meaningful_authority_count),
    }
    summary = {
        "has_explicit_ref": 1 if has_explicit_ref else 0,
        "has_any_source_evidence_signal": 1 if has_any_signal else 0,
        "has_source_url_flag": 1 if has_source_url_flag else 0,
        "has_source_media_flag": 1 if has_source_media_flag else 0,
        "n_refs_total": n_refs_total,
        "n_url_refs": n_url_refs,
        "n_media_refs": n_media_refs,
        "n_reference_types_present": n_reference_types_present,
        "has_url_ref_type": has_url_ref_type,
        "has_media_ref_type": has_media_ref_type,
        "has_multi_ref_type": has_multi_ref_type,
        "has_multi_ref_rows": has_multi_ref_rows,
        "has_normalized_reference": has_normalized_reference,
        "has_domain": has_domain,
        "n_refs_with_normalized_value": n_refs_with_normalized_value,
        "n_refs_with_domain": n_refs_with_domain,
        "has_target_post_id": has_target_post_id,
        "n_target_post_id_refs": n_target_post_id_refs,
        "has_specific_reference_signal": has_specific_reference_signal,
        "n_internal_refs": n_internal_refs,
        "n_external_refs": n_external_refs,
        "has_internal_ref": has_internal_ref,
        "has_external_ref": has_external_ref,
        "authority_hist_unknown": authority_counts.get("unknown", 0),
        "authority_hist_internal_platform": authority_counts.get("internal_platform", 0),
        "authority_hist_media_cdn_or_image_host": authority_counts.get("media_cdn_or_image_host", 0),
        "authority_hist_shortlink": authority_counts.get("shortlink", 0),
        "authority_hist_social_or_platform_hosted_external": authority_counts.get("social_or_platform_hosted_external", 0),
        "authority_hist_general_external": authority_counts.get("general_external", 0),
    }
    summary_vector = np.asarray([float(summary[name]) for name in config.summary_feature_names], dtype=np.float32)
    quality_vector = np.asarray([int(quality[name]) for name in config.summary_quality_flag_names], dtype=np.uint8)
    return {"summary": summary, "quality": quality}, summary_vector, quality_vector
