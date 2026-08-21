from __future__ import annotations

from models.encoders.evidence_view.config import EvidenceViewConfig


def normalize_source_channel(config: EvidenceViewConfig, source: str | None) -> str:
    if not source:
        return "unknown"
    canonical = config.source_channel_aliases.get(source, source)
    return canonical if canonical in config.source_channel_vocab else "unknown"


def source_channel_id(config: EvidenceViewConfig, source: str | None) -> int:
    return int(config.source_channel_vocab[normalize_source_channel(config, source)])


def bucket_authority(
    config: EvidenceViewConfig,
    domain: str | None,
    is_internal_platform_ref: bool | None,
    ref_type: str | None,
    normalized_value: str | None,
) -> tuple[int, str]:
    domain_norm = (domain or "").strip().lower()
    if is_internal_platform_ref is True:
        return config.authority_bucket_vocab["internal_platform"], "internal_platform"
    if domain_norm in {item.lower() for item in config.shortlink_domains}:
        return config.authority_bucket_vocab["shortlink"], "shortlink"
    if domain_norm in {item.lower() for item in config.media_domains} or (ref_type or "") == "media":
        return config.authority_bucket_vocab["media_cdn_or_image_host"], "media_cdn_or_image_host"
    if domain_norm in {item.lower() for item in config.platform_domains}:
        return config.authority_bucket_vocab["social_or_platform_hosted_external"], "social_or_platform_hosted_external"
    if domain_norm or normalized_value:
        return config.authority_bucket_vocab["general_external"], "general_external"
    return config.authority_bucket_vocab["unknown"], "unknown"
