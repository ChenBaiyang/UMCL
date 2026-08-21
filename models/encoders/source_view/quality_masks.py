from __future__ import annotations


def build_quality_masks(profile: dict, observed_counts: dict[str, int]) -> dict[str, int]:
    platform_extra = profile.get("platform_extra") or {}
    missing = {
        "missing_verified": int(profile.get("verified") is None),
        "missing_account_age_days": int(profile.get("account_age_days") is None),
        "missing_profile_description": int((profile.get("profile_description") or {}).get("raw") in (None, "")),
        "missing_display_name": int((profile.get("display_name") or {}).get("raw") in (None, "")),
        "missing_followers_count": int(profile.get("followers_count") is None),
        "missing_following_count": int(profile.get("following_count") is None),
        "missing_post_count": int(profile.get("post_count") is None),
        "missing_listed_count": int(platform_extra.get("listed_count") is None),
        "missing_verified_reason": int(platform_extra.get("verified_reason") in (None, "")),
        "missing_verified_type": int(platform_extra.get("verified_type") is None),
        "missing_gender": int(platform_extra.get("gender") in (None, "")),
        "missing_location": int((platform_extra.get("location") in (None, "")) and (platform_extra.get("user_location") in (None, ""))),
        "missing_avatar_url": int(platform_extra.get("user_avatar") in (None, "")),
        "missing_favourites_count": int(platform_extra.get("favourites_count") is None),
        "missing_geo_enabled": int(platform_extra.get("user_geo_enabled") is None),
        "missing_membership_indicator": int(platform_extra.get("bi_followers_count") is None),
        "missing_follower_followee_ratio": int((profile.get("followers_count") is None) or (profile.get("following_count") in (None, 0))),
        "missing_lifetime_posting_rate_proxy": int((profile.get("account_age_days") is None) or (profile.get("account_age_days") in (0, 0.0)) or (profile.get("post_count") is None)),
    }
    source_signal_observed = int(bool(profile))
    identity_sparse = int(observed_counts["identity"] <= 2)
    social_sparse = int(observed_counts["social"] <= 1)
    role_sparse = int(observed_counts["role"] <= 1)
    history_sparse = int(observed_counts["history"] == 0)
    source_sparse = int(sum(observed_counts.values()) <= 3)
    return {
        "source_signal_observed": source_signal_observed,
        "source_sparse_mask": source_sparse,
        "source_identity_sparse_mask": identity_sparse,
        "source_social_sparse_mask": social_sparse,
        "source_role_sparse_mask": role_sparse,
        "source_history_sparse_mask": history_sparse,
        **missing,
    }
