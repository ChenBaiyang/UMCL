from __future__ import annotations

from typing import Any

import numpy as np

from models.encoders.source_view.config import SourceViewConfig


def _feature_index_map(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def _observed_mask(
    config: SourceViewConfig,
    feature_name: str,
    quality_flags: np.ndarray,
    quality_index_by_name: dict[str, int],
) -> np.ndarray:
    missing_flag_name = config.summary_feature_missing_flag_names.get(feature_name)
    if missing_flag_name is None:
        return np.ones(quality_flags.shape[0], dtype=bool)
    flag_idx = quality_index_by_name.get(missing_flag_name)
    if flag_idx is None:
        return np.ones(quality_flags.shape[0], dtype=bool)
    return quality_flags[:, flag_idx] == 0


def build_summary_normalization_stats(
    config: SourceViewConfig,
    summary_features: np.ndarray,
    summary_quality_flags: np.ndarray,
    canonical_splits: list[str],
) -> dict[str, dict[str, Any]]:
    summary_index_by_name = _feature_index_map(config.summary_feature_names)
    quality_index_by_name = _feature_index_map(config.summary_quality_flag_names)
    train_split_mask = np.asarray([split == "train" for split in canonical_splits], dtype=bool)
    stats: dict[str, dict[str, Any]] = {}

    for feature_name in config.normalized_summary_feature_names:
        feature_idx = summary_index_by_name.get(feature_name)
        if feature_idx is None:
            continue
        observed_mask = _observed_mask(config, feature_name, summary_quality_flags, quality_index_by_name)
        train_observed_mask = train_split_mask & observed_mask
        raw_values = summary_features[train_observed_mask, feature_idx].astype(np.float64, copy=False)
        if raw_values.size == 0:
            mean = 0.0
            std = 1.0
        else:
            transformed = np.log1p(np.clip(raw_values, a_min=0.0, a_max=None))
            mean = float(np.mean(transformed))
            std = float(np.std(transformed))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
        stats[feature_name] = {
            "transform": "log1p_zscore",
            "mean": mean,
            "std": std,
            "observed_train_count": int(raw_values.size),
        }
    return stats


def apply_summary_normalization(
    config: SourceViewConfig,
    summary_features: np.ndarray,
    summary_quality_flags: np.ndarray,
    normalization_stats: dict[str, dict[str, Any]],
) -> np.ndarray:
    normalized = np.asarray(summary_features, dtype=np.float32).copy()
    summary_index_by_name = _feature_index_map(config.summary_feature_names)
    quality_index_by_name = _feature_index_map(config.summary_quality_flag_names)

    for feature_name, feature_stats in normalization_stats.items():
        feature_idx = summary_index_by_name.get(feature_name)
        if feature_idx is None:
            continue
        observed_mask = _observed_mask(config, feature_name, summary_quality_flags, quality_index_by_name)
        raw_column = normalized[:, feature_idx].astype(np.float64, copy=False)
        transformed = np.log1p(np.clip(raw_column, a_min=0.0, a_max=None))
        standardized = (transformed - float(feature_stats["mean"])) / float(feature_stats["std"])
        normalized[:, feature_idx] = np.where(observed_mask, standardized, 0.0).astype(np.float32, copy=False)
    return normalized
