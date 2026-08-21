from __future__ import annotations

from typing import Any

import numpy as np

from models.encoders.source_view.config import SourceViewConfig
from models.encoders.source_view.quality_masks import build_quality_masks
from models.encoders.source_view.role_builder import build_role_signals


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _bucket_age(config: SourceViewConfig, account_age_days: Any) -> int:
    if account_age_days in (None, ""):
        return config.age_bucket_vocab["unknown"]
    age = float(account_age_days)
    if age < 30:
        return config.age_bucket_vocab["new"]
    if age < 180:
        return config.age_bucket_vocab["young"]
    if age < 730:
        return config.age_bucket_vocab["mature"]
    return config.age_bucket_vocab["established"]


def _bucket_activity(config: SourceViewConfig, rate: Any) -> int:
    if rate in (None, ""):
        return config.activity_rate_bucket_vocab["unknown"]
    r = float(rate)
    if r < 1:
        return config.activity_rate_bucket_vocab["low"]
    if r < 10:
        return config.activity_rate_bucket_vocab["moderate"]
    if r < 50:
        return config.activity_rate_bucket_vocab["active"]
    return config.activity_rate_bucket_vocab["very_active"]


def build_source_summary(config: SourceViewConfig, event_row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    source = event_row.get("source") or {}
    profile = source.get("profile") or {}
    platform_extra = profile.get("platform_extra") or {}
    profile_description = profile.get("profile_description") or {}
    display_name = profile.get("display_name") or {}

    verified_flag = float(profile.get("verified") is True)
    has_account_created_at = float(profile.get("account_created_at") is not None)
    account_age_days = _safe_float(profile.get("account_age_days"))
    has_profile_description = float(profile_description.get("raw") not in (None, ""))
    has_display_name = float(display_name.get("raw") not in (None, ""))
    profile_description_length_chars = float(profile_description.get("char_len") or len(profile_description.get("raw") or ""))
    display_name_length_chars = float(display_name.get("char_len") or len(display_name.get("raw") or ""))
    avatar_present_flag = float(platform_extra.get("user_avatar") not in (None, ""))
    location_present_flag = float((platform_extra.get("location") not in (None, "")) or (platform_extra.get("user_location") not in (None, "")))
    gender_present_flag = float(platform_extra.get("gender") not in (None, ""))
    verified_reason_present_flag = float(platform_extra.get("verified_reason") not in (None, ""))
    verified_type_present_flag = float(platform_extra.get("verified_type") is not None)
    profile_completeness_count = (
        has_account_created_at
        + has_profile_description
        + has_display_name
        + avatar_present_flag
        + location_present_flag
        + gender_present_flag
        + verified_reason_present_flag
        + verified_type_present_flag
    )

    followers_count_raw = _safe_float(profile.get("followers_count"))
    following_count_raw = _safe_float(profile.get("following_count"))
    if profile.get("followers_count") is not None and profile.get("following_count") not in (None, 0):
        follower_followee_ratio = float(profile.get("followers_count")) / float(profile.get("following_count"))
    else:
        follower_followee_ratio = 0.0
    post_count_raw = _safe_float(profile.get("post_count"))
    listed_count_raw = _safe_float(platform_extra.get("listed_count"))
    membership_indicator_bi_followers_raw = _safe_float(platform_extra.get("bi_followers_count"))
    favourites_count_raw = _safe_float(platform_extra.get("favourites_count"))
    geo_enabled_flag = float(bool(platform_extra.get("user_geo_enabled"))) if platform_extra.get("user_geo_enabled") is not None else 0.0

    if account_age_days > 0 and profile.get("post_count") is not None:
        lifetime_posting_rate_proxy = post_count_raw / account_age_days
    else:
        lifetime_posting_rate_proxy = 0.0
    age_bucket_id = _bucket_age(config, profile.get("account_age_days"))
    activity_rate_bucket_id = _bucket_activity(config, lifetime_posting_rate_proxy if lifetime_posting_rate_proxy > 0 else None)

    role_signals = build_role_signals(config, profile)

    observed_counts = {
        "identity": int(
            sum(
                [
                    has_account_created_at,
                    has_profile_description,
                    has_display_name,
                    avatar_present_flag,
                    location_present_flag,
                    gender_present_flag,
                    verified_reason_present_flag,
                    verified_type_present_flag,
                    float(profile.get("verified") is not None),
                    float(profile.get("account_age_days") is not None),
                ]
            )
        ),
        "social": int(
            sum(
                [
                    float(profile.get("followers_count") is not None),
                    float(profile.get("following_count") is not None),
                    float(profile.get("post_count") is not None),
                    float(platform_extra.get("listed_count") is not None),
                    float(platform_extra.get("bi_followers_count") is not None),
                    float(platform_extra.get("favourites_count") is not None),
                    float(platform_extra.get("user_geo_enabled") is not None),
                ]
            )
        ),
        "role": int(role_signals.explicit_role_present + role_signals.profile_description_has_role_like_tokens + role_signals.display_name_has_org_like_tokens),
        "history": int(float(profile.get("account_age_days") is not None) + float(lifetime_posting_rate_proxy > 0)),
    }
    quality = build_quality_masks(profile, observed_counts)
    source_mode = "missing"
    if quality["source_signal_observed"]:
        source_mode = "minimal" if quality["source_sparse_mask"] else "observed"

    summary_payload = {
        "verified_flag": verified_flag,
        "has_account_created_at": has_account_created_at,
        "account_age_days": account_age_days,
        "has_profile_description": has_profile_description,
        "has_display_name": has_display_name,
        "profile_description_length_chars": profile_description_length_chars,
        "display_name_length_chars": display_name_length_chars,
        "profile_completeness_count": float(profile_completeness_count),
        "avatar_present_flag": avatar_present_flag,
        "location_present_flag": location_present_flag,
        "gender_present_flag": gender_present_flag,
        "verified_reason_present_flag": verified_reason_present_flag,
        "verified_type_present_flag": verified_type_present_flag,
        "display_name_has_org_like_tokens": role_signals.display_name_has_org_like_tokens,
        "profile_description_has_role_like_tokens": role_signals.profile_description_has_role_like_tokens,
        "name_description_role_consistency_flag": role_signals.name_description_role_consistency_flag,
        "followers_count_raw": followers_count_raw,
        "following_count_raw": following_count_raw,
        "follower_followee_ratio": float(follower_followee_ratio),
        "post_count_raw": post_count_raw,
        "listed_count_raw": listed_count_raw,
        "membership_indicator_bi_followers_raw": membership_indicator_bi_followers_raw,
        "favourites_count_raw": favourites_count_raw,
        "geo_enabled_flag": geo_enabled_flag,
        "lifetime_posting_rate_proxy": float(lifetime_posting_rate_proxy),
        "explicit_role_present": role_signals.explicit_role_present,
        "verified_reason_present": role_signals.verified_reason_present,
        "role_flag_official": role_signals.role_flag_official,
        "role_flag_media": role_signals.role_flag_media,
        "role_flag_brand": role_signals.role_flag_brand,
        "role_flag_public_figure": role_signals.role_flag_public_figure,
        "role_flag_general_user": role_signals.role_flag_general_user,
    }
    summary_vector = np.asarray([summary_payload[name] for name in config.summary_feature_names], dtype=np.float32)
    quality_vector = np.asarray([quality[name] for name in config.summary_quality_flag_names], dtype=np.uint8)
    categorical_vector = np.asarray(
        [role_signals.verified_type_id, role_signals.role_bucket_id, age_bucket_id, activity_rate_bucket_id],
        dtype=np.int16,
    )
    payload = {
        "source_mode": source_mode,
        "summary": summary_payload,
        "quality": quality,
        "categorical": {
            "verified_type_id": role_signals.verified_type_id,
            "role_bucket_id": role_signals.role_bucket_id,
            "age_bucket_id": age_bucket_id,
            "activity_rate_bucket_id": activity_rate_bucket_id,
        },
    }
    return payload, summary_vector, quality_vector, categorical_vector
