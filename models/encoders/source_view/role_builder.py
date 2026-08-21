from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.encoders.source_view.config import SourceViewConfig


@dataclass
class RoleSignals:
    explicit_role_present: float
    verified_reason_present: float
    verified_type_id: int
    role_bucket_id: int
    role_flag_official: float
    role_flag_media: float
    role_flag_brand: float
    role_flag_public_figure: float
    role_flag_general_user: float
    display_name_has_org_like_tokens: float
    profile_description_has_role_like_tokens: float
    name_description_role_consistency_flag: float


def _contains_any(text: str, tokens: list[str]) -> bool:
    text_lower = text.lower()
    return any(token.lower() in text_lower for token in tokens)


def _role_bucket_from_verified_reason(config: SourceViewConfig, verified_reason: str) -> str:
    if not verified_reason:
        return "unknown"
    if _contains_any(verified_reason, config.role_media_tokens_en + config.role_media_tokens_zh):
        return "news_or_media"
    if _contains_any(verified_reason, config.org_like_tokens_en + config.org_like_tokens_zh):
        return "official_org_or_institution"
    if _contains_any(verified_reason, config.role_brand_tokens_en + config.role_brand_tokens_zh):
        return "company_or_brand"
    if _contains_any(verified_reason, config.role_public_figure_tokens_en + config.role_public_figure_tokens_zh):
        return "public_figure_or_celebrity"
    return "individual_or_general_user"


def _role_bucket_from_text(config: SourceViewConfig, text: str) -> str:
    if not text:
        return "unknown"
    if _contains_any(text, config.role_media_tokens_en + config.role_media_tokens_zh):
        return "news_or_media"
    if _contains_any(text, config.org_like_tokens_en + config.org_like_tokens_zh):
        return "official_org_or_institution"
    if _contains_any(text, config.role_brand_tokens_en + config.role_brand_tokens_zh):
        return "company_or_brand"
    if _contains_any(text, config.role_public_figure_tokens_en + config.role_public_figure_tokens_zh):
        return "public_figure_or_celebrity"
    return "individual_or_general_user"


def map_verified_type(config: SourceViewConfig, verified: Any, verified_type: Any) -> int:
    if verified_type is None:
        if verified is True:
            return config.verified_type_vocab["platform_verified"]
        if verified is False:
            return config.verified_type_vocab["unverified"]
        return config.verified_type_vocab["unknown"]
    try:
        numeric = int(verified_type)
    except Exception:
        numeric = None
    if numeric is None:
        return config.verified_type_vocab["other_verified"]
    if numeric < 0:
        return config.verified_type_vocab["unverified"]
    if numeric == 0:
        return config.verified_type_vocab["platform_verified"]
    if numeric > 0:
        return config.verified_type_vocab["organization_verified"]
    return config.verified_type_vocab["other_verified"]


def build_role_signals(config: SourceViewConfig, profile: dict[str, Any]) -> RoleSignals:
    platform_extra = profile.get("platform_extra") or {}
    verified_reason = str(platform_extra.get("verified_reason") or "").strip()
    display_name = str((profile.get("display_name") or {}).get("norm_light") or (profile.get("display_name") or {}).get("raw") or "").strip()
    profile_desc = str((profile.get("profile_description") or {}).get("norm_light") or (profile.get("profile_description") or {}).get("raw") or "").strip()
    verified_type_id = map_verified_type(config, profile.get("verified"), platform_extra.get("verified_type"))

    explicit_bucket = _role_bucket_from_verified_reason(config, verified_reason)
    display_bucket = _role_bucket_from_text(config, display_name) if config.use_profile_text_cues else "unknown"
    desc_bucket = _role_bucket_from_text(config, profile_desc) if config.use_profile_text_cues else "unknown"

    role_bucket = explicit_bucket
    if role_bucket == "unknown":
        for candidate in (desc_bucket, display_bucket):
            if candidate != "unknown":
                role_bucket = candidate
                break
    if role_bucket == "unknown" and (display_name or profile_desc):
        role_bucket = "individual_or_general_user"

    name_org_like = float(_contains_any(display_name, config.org_like_tokens_en + config.org_like_tokens_zh + config.role_brand_tokens_en + config.role_brand_tokens_zh))
    desc_role_like = float(
        _contains_any(
            profile_desc,
            config.org_like_tokens_en
            + config.org_like_tokens_zh
            + config.role_media_tokens_en
            + config.role_media_tokens_zh
            + config.role_brand_tokens_en
            + config.role_brand_tokens_zh
            + config.role_public_figure_tokens_en
            + config.role_public_figure_tokens_zh,
        )
    )
    consistency = float(display_bucket != "unknown" and desc_bucket != "unknown" and display_bucket == desc_bucket)

    return RoleSignals(
        explicit_role_present=float(bool(verified_reason or platform_extra.get("verified_type") not in (None, ""))),
        verified_reason_present=float(bool(verified_reason)),
        verified_type_id=verified_type_id,
        role_bucket_id=config.role_bucket_vocab[role_bucket],
        role_flag_official=float(role_bucket == "official_org_or_institution"),
        role_flag_media=float(role_bucket == "news_or_media"),
        role_flag_brand=float(role_bucket == "company_or_brand"),
        role_flag_public_figure=float(role_bucket == "public_figure_or_celebrity"),
        role_flag_general_user=float(role_bucket == "individual_or_general_user"),
        display_name_has_org_like_tokens=name_org_like,
        profile_description_has_role_like_tokens=desc_role_like,
        name_description_role_consistency_flag=consistency,
    )
