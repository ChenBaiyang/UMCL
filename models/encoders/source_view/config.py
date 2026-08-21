from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["lite", "full"]


@dataclass
class SourceViewConfig:
    dataset_id: str | None = None
    normalize_summary_features: bool = True
    use_profile_text_cues: bool = True
    use_verified_reason: bool = True
    use_verified_type: bool = True
    use_account_age_days: bool = True
    use_lifetime_posting_rate_proxy: bool = True
    build_second_temporal_pass: bool = False
    storage_backend: str = "npz"
    shard_size: int = 4096
    role_bucket_vocab: dict[str, int] = field(
        default_factory=lambda: {
            "unknown": 0,
            "official_org_or_institution": 1,
            "news_or_media": 2,
            "company_or_brand": 3,
            "public_figure_or_celebrity": 4,
            "individual_or_general_user": 5,
        }
    )
    verified_type_vocab: dict[str, int] = field(
        default_factory=lambda: {
            "unknown": 0,
            "unverified": 1,
            "platform_verified": 2,
            "organization_verified": 3,
            "other_verified": 4,
        }
    )
    age_bucket_vocab: dict[str, int] = field(
        default_factory=lambda: {"unknown": 0, "new": 1, "young": 2, "mature": 3, "established": 4}
    )
    activity_rate_bucket_vocab: dict[str, int] = field(
        default_factory=lambda: {"unknown": 0, "low": 1, "moderate": 2, "active": 3, "very_active": 4}
    )

    summary_feature_names: list[str] = field(
        default_factory=lambda: [
            "verified_flag",
            "has_account_created_at",
            "account_age_days",
            "has_profile_description",
            "has_display_name",
            "profile_description_length_chars",
            "display_name_length_chars",
            "profile_completeness_count",
            "avatar_present_flag",
            "location_present_flag",
            "gender_present_flag",
            "verified_reason_present_flag",
            "verified_type_present_flag",
            "display_name_has_org_like_tokens",
            "profile_description_has_role_like_tokens",
            "name_description_role_consistency_flag",
            "followers_count_raw",
            "following_count_raw",
            "follower_followee_ratio",
            "post_count_raw",
            "listed_count_raw",
            "membership_indicator_bi_followers_raw",
            "favourites_count_raw",
            "geo_enabled_flag",
            "lifetime_posting_rate_proxy",
            "explicit_role_present",
            "verified_reason_present",
            "role_flag_official",
            "role_flag_media",
            "role_flag_brand",
            "role_flag_public_figure",
            "role_flag_general_user",
        ]
    )
    summary_quality_flag_names: list[str] = field(
        default_factory=lambda: [
            "source_signal_observed",
            "source_sparse_mask",
            "source_identity_sparse_mask",
            "source_social_sparse_mask",
            "source_role_sparse_mask",
            "source_history_sparse_mask",
            "missing_verified",
            "missing_account_age_days",
            "missing_profile_description",
            "missing_display_name",
            "missing_followers_count",
            "missing_following_count",
            "missing_post_count",
            "missing_listed_count",
            "missing_verified_reason",
            "missing_verified_type",
            "missing_gender",
            "missing_location",
            "missing_avatar_url",
            "missing_favourites_count",
            "missing_geo_enabled",
            "missing_membership_indicator",
            "missing_follower_followee_ratio",
            "missing_lifetime_posting_rate_proxy",
        ]
    )
    categorical_feature_names: list[str] = field(
        default_factory=lambda: ["verified_type_id", "role_bucket_id", "age_bucket_id", "activity_rate_bucket_id"]
    )
    normalized_summary_feature_names: list[str] = field(
        default_factory=lambda: [
            "account_age_days",
            "profile_description_length_chars",
            "display_name_length_chars",
            "followers_count_raw",
            "following_count_raw",
            "follower_followee_ratio",
            "post_count_raw",
            "listed_count_raw",
            "membership_indicator_bi_followers_raw",
            "favourites_count_raw",
            "lifetime_posting_rate_proxy",
        ]
    )
    summary_feature_missing_flag_names: dict[str, str] = field(
        default_factory=lambda: {
            "account_age_days": "missing_account_age_days",
            "profile_description_length_chars": "missing_profile_description",
            "display_name_length_chars": "missing_display_name",
            "followers_count_raw": "missing_followers_count",
            "following_count_raw": "missing_following_count",
            "follower_followee_ratio": "missing_follower_followee_ratio",
            "post_count_raw": "missing_post_count",
            "listed_count_raw": "missing_listed_count",
            "membership_indicator_bi_followers_raw": "missing_membership_indicator",
            "favourites_count_raw": "missing_favourites_count",
            "lifetime_posting_rate_proxy": "missing_lifetime_posting_rate_proxy",
        }
    )

    org_like_tokens_en: list[str] = field(
        default_factory=lambda: ["official", "gov", "government", "dept", "department", "agency", "university", "college", "hospital", "committee", "foundation", "ministry", "police"]
    )
    org_like_tokens_zh: list[str] = field(default_factory=list)
    role_media_tokens_en: list[str] = field(
        default_factory=lambda: ["news", "journalist", "reporter", "editor", "media", "radio", "tv", "press", "bbc", "cnn", "times", "post"]
    )
    role_media_tokens_zh: list[str] = field(default_factory=list)
    role_brand_tokens_en: list[str] = field(
        default_factory=lambda: ["company", "brand", "inc", "corp", "ltd", "studio", "shop", "official store", "group"]
    )
    role_brand_tokens_zh: list[str] = field(default_factory=list)
    role_public_figure_tokens_en: list[str] = field(
        default_factory=lambda: ["actor", "actress", "singer", "artist", "writer", "author", "host", "anchor", "celebrity", "chairman", "ceo"]
    )
    role_public_figure_tokens_zh: list[str] = field(default_factory=list)

    event_hidden_dim: int = 128
    summary_mlp_hidden_dim: int = 128
    summary_branch_hidden_dim: int = 128
    categorical_feature_embedding_dim: int = 16
    categorical_mlp_hidden_dim: int = 64
    categorical_branch_hidden_dim: int = 128
    summary_categorical_fusion_hidden_dim: int = 128
    summary_linear_shortcut: bool = True
    dropout: float = 0.1
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def event_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.event_hidden_dim)

    def summary_branch_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_branch_hidden_dim)

    def categorical_branch_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.categorical_branch_hidden_dim)

    def fusion_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_categorical_fusion_hidden_dim)

    def summary_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_mlp_hidden_dim)

    def cat_emb_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.categorical_feature_embedding_dim)

    def cat_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.categorical_mlp_hidden_dim)

    def artifact_mode(self, mode: Mode) -> Literal["lite", "full"]:
        return "full" if mode == "full" else "lite"

    def as_dict(self) -> dict:
        return asdict(self)
