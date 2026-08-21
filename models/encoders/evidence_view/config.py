from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["lite", "full"]


@dataclass
class EvidenceViewConfig:
    dataset_id: str | None = None
    domain_bucket_policy: str = "coarse"
    storage_backend: str = "npz"
    shard_size: int = 4096
    set_pooling: str = "masked_mean"

    ref_type_vocab: dict[str, int] = field(
        default_factory=lambda: {"unknown": 0, "url": 1, "media": 2, "quoted_post": 3, "reposted_post": 4}
    )
    authority_bucket_vocab: dict[str, int] = field(
        default_factory=lambda: {
            "unknown": 0,
            "internal_platform": 1,
            "media_cdn_or_image_host": 2,
            "shortlink": 3,
            "social_or_platform_hosted_external": 4,
            "general_external": 5,
        }
    )
    source_channel_vocab: dict[str, int] = field(
        default_factory=lambda: {"unknown": 0, "entities.urls": 1, "entities.media": 2, "text_raw": 3, "payload.url": 4, "preview.images": 5}
    )
    source_channel_aliases: dict[str, str] = field(
        default_factory=lambda: {"text": "text_raw", "original_text": "text_raw", "picture": "entities.media"}
    )
    shortlink_domains: list[str] = field(
        default_factory=lambda: ["t.cn", "bit.ly", "bbc.in", "cnn.it", "fxn.ws", "trib.al", "on.rt.com", "ow.ly", "ind.pn", "on.wsj.com"]
    )
    media_domains: list[str] = field(
        default_factory=lambda: ["pbs.twimg.com", "i.redditmedia.com", "ww1.sinaimg.cn", "ww2.sinaimg.cn", "ww3.sinaimg.cn", "ww4.sinaimg.cn"]
    )
    platform_domains: list[str] = field(default_factory=lambda: ["twitter.com", "www.reddit.com", "weibo.com"])

    summary_feature_names: list[str] = field(
        default_factory=lambda: [
            "has_explicit_ref",
            "has_any_source_evidence_signal",
            "has_source_url_flag",
            "has_source_media_flag",
            "n_refs_total",
            "n_url_refs",
            "n_media_refs",
            "n_reference_types_present",
            "has_url_ref_type",
            "has_media_ref_type",
            "has_multi_ref_type",
            "has_multi_ref_rows",
            "has_normalized_reference",
            "has_domain",
            "n_refs_with_normalized_value",
            "n_refs_with_domain",
            "has_target_post_id",
            "n_target_post_id_refs",
            "has_specific_reference_signal",
            "n_internal_refs",
            "n_external_refs",
            "has_internal_ref",
            "has_external_ref",
            "authority_hist_unknown",
            "authority_hist_internal_platform",
            "authority_hist_media_cdn_or_image_host",
            "authority_hist_shortlink",
            "authority_hist_social_or_platform_hosted_external",
            "authority_hist_general_external",
        ]
    )
    summary_quality_flag_names: list[str] = field(
        default_factory=lambda: [
            "evidence_signal_observed",
            "no_reference_rows_mask",
            "weak_evidence_mask",
            "sparse_evidence_mask",
            "authority_unknown_mask",
        ]
    )

    event_hidden_dim: int = 128
    summary_mlp_hidden_dim: int = 128
    summary_branch_hidden_dim: int = 128
    reference_category_embedding_dim: int = 16
    reference_mlp_hidden_dim: int = 96
    reference_branch_hidden_dim: int = 128
    summary_reference_fusion_hidden_dim: int = 128
    reference_attention_pooling_hidden_dim: int = 64
    dropout: float = 0.1
    resolved_overrides: dict[str, Any] = field(default_factory=dict)

    def event_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.event_hidden_dim)

    def summary_branch_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_branch_hidden_dim)

    def reference_branch_hidden_dim_value(self, mode: Mode) -> int:
        _ = mode
        return int(self.reference_branch_hidden_dim)

    def fusion_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_reference_fusion_hidden_dim)

    def ref_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.reference_mlp_hidden_dim)

    def summary_hidden_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.summary_mlp_hidden_dim)

    def ref_embedding_dim(self, mode: Mode) -> int:
        _ = mode
        return int(self.reference_category_embedding_dim)

    def artifact_mode(self, mode: Mode) -> Literal["lite", "full"]:
        return "full" if mode == "full" else "lite"

    def as_dict(self) -> dict:
        return asdict(self)
