from __future__ import annotations


def evidence_signal_observed(has_explicit_ref: bool, has_source_url_flag: bool, has_source_media_flag: bool) -> int:
    return 1 if (has_explicit_ref or has_source_url_flag or has_source_media_flag) else 0


def no_reference_rows_mask(n_refs_total: int) -> int:
    return 1 if n_refs_total == 0 else 0


def weak_evidence_mask(has_any_signal: bool, n_refs_total: int) -> int:
    return 1 if has_any_signal and n_refs_total == 0 else 0


def sparse_evidence_mask(n_refs_total: int, has_multi_ref_type: bool, has_multi_ref_rows: bool) -> int:
    if n_refs_total == 0:
        return 1
    if n_refs_total == 1 and not has_multi_ref_type and not has_multi_ref_rows:
        return 1
    return 0


def authority_unknown_mask(n_refs_total: int, meaningful_authority_count: int) -> int:
    if n_refs_total == 0:
        return 1
    return 1 if meaningful_authority_count == 0 else 0
