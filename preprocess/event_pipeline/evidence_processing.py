from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from preprocess.event_pipeline.schema import EvidenceSummary, EvidenceView


INTERNAL_DOMAINS = {
    "twitter": {"twitter.com", "t.co", "x.com"},
    "reddit": {"reddit.com", "redd.it"},
    "weibo": {"weibo.com", "t.cn"},
}


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return value
    parsed = urlparse(value)
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True) if not key.lower().startswith("utm_")]
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path, parsed.params, urlencode(query), ""))


def preprocess_evidence_view(view: EvidenceView, platform: str) -> EvidenceView:
    dedup = {}
    processed_refs = []
    internal_domains = INTERNAL_DOMAINS.get(platform, set())
    for ref in view.references:
        ref.normalized_value = _normalize_url(ref.normalized_value or ref.raw_value)
        if ref.normalized_value and not ref.domain:
            try:
                ref.domain = urlparse(ref.normalized_value).netloc.lower() or None
            except ValueError:
                ref.domain = None
        if ref.ref_type in {"quoted_post", "reposted_post"}:
            ref.is_internal_platform_ref = True
        elif ref.domain is not None:
            ref.is_internal_platform_ref = ref.domain in internal_domains
        key = (ref.ref_type, ref.normalized_value, ref.target_post_id or "")
        if key in dedup:
            continue
        dedup[key] = True
        processed_refs.append(ref)

    view.references = sorted(processed_refs, key=lambda item: item.ref_id)
    view.summary = EvidenceSummary(
        scope="source_only",
        has_explicit_ref=bool(view.references),
        n_refs_total=len(view.references),
        n_url_refs=sum(1 for ref in view.references if ref.ref_type == "url"),
        n_media_refs=sum(1 for ref in view.references if ref.ref_type == "media"),
        n_internal_refs=sum(1 for ref in view.references if ref.is_internal_platform_ref is True),
        n_external_refs=sum(1 for ref in view.references if ref.is_internal_platform_ref is False),
        thread_n_refs_total=view.summary.thread_n_refs_total,
    )
    return view
