"""Reporting regressions for the mixed-subset experiment."""

from __future__ import annotations

import unittest

from experiment_mixed_subsets import (
    DEFAULT_DATASETS,
    DEFAULT_SEEDS,
    REGIME_LABELS,
    SUBSET_ORDER,
    _aggregate_rows,
    _gap_rows,
    _latex_table,
    _summary_rows,
    _validate_complete_grid,
    _validate_seed_grid,
)


class UnifiedMixedSubsetReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_rows = []
        for dataset_index, dataset_id in enumerate(DEFAULT_DATASETS):
            for regime_index, regime in enumerate(REGIME_LABELS):
                for seed in DEFAULT_SEEDS:
                    for subset_index, subset in enumerate(SUBSET_ORDER):
                        base = 0.60 + dataset_index * 0.10 + regime_index * 0.01 + subset_index * 0.001
                        self.seed_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "regime": regime,
                                "seed": seed,
                                "subset": subset,
                                "macro_f1": base + seed * 1e-5,
                                "accuracy": base + 0.02 + seed * 1e-5,
                            }
                        )

    def test_aggregate_contains_mean_sample_std_and_five_seeds(self) -> None:
        aggregate = _aggregate_rows(self.seed_rows, list(DEFAULT_DATASETS))
        expected_count = len(DEFAULT_DATASETS) * len(REGIME_LABELS) * len(SUBSET_ORDER)
        self.assertEqual(len(aggregate), expected_count)
        self.assertTrue(all(row["macro_f1_seed_count"] == 5 for row in aggregate))
        self.assertTrue(all(row["macro_f1_sample_std"] is not None for row in aggregate))
        _validate_seed_grid(self.seed_rows, list(DEFAULT_DATASETS), list(DEFAULT_SEEDS))
        _validate_complete_grid(aggregate, list(DEFAULT_DATASETS), list(DEFAULT_SEEDS))

    def test_seed_grid_rejects_a_duplicate_and_missing_seed(self) -> None:
        broken = list(self.seed_rows)
        target = next(
            index for index, row in enumerate(broken)
            if row["dataset_id"] == "PHEME"
            and row["regime"] == "mixed"
            and row["subset"] == "C"
            and row["seed"] == 32
        )
        broken[target] = {**broken[target], "seed": 16}
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
            _validate_seed_grid(broken, list(DEFAULT_DATASETS), list(DEFAULT_SEEDS))

    def test_summary_is_unweighted_over_seven_subsets(self) -> None:
        aggregate = _aggregate_rows(self.seed_rows, list(DEFAULT_DATASETS))
        summaries = _summary_rows(aggregate, list(DEFAULT_DATASETS))
        self.assertEqual(len(summaries), len(DEFAULT_DATASETS) * len(REGIME_LABELS))
        self.assertTrue(all(row["subset_count"] == 7 for row in summaries))

    def test_gap_is_unified_minus_specialist_and_keeps_negative_values(self) -> None:
        aggregate = _aggregate_rows(self.seed_rows, list(DEFAULT_DATASETS))
        specialist = next(
            row for row in aggregate
            if row["dataset_id"] == "PHEME" and row["regime"] == "subset_specific" and row["subset"] == "C"
        )
        unified = next(
            row for row in aggregate
            if row["dataset_id"] == "PHEME" and row["regime"] == "mixed" and row["subset"] == "C"
        )
        unified["macro_f1_mean"] = float(specialist["macro_f1_mean"]) - 0.05
        gaps = _gap_rows(aggregate, list(DEFAULT_DATASETS))
        self.assertEqual(len(gaps), len(DEFAULT_DATASETS) * len(SUBSET_ORDER))
        negative = next(row for row in gaps if row["dataset_id"] == "PHEME" and row["subset"] == "C")
        self.assertAlmostEqual(negative["signed_gap_unified_minus_specialist"], -0.05)
        self.assertAlmostEqual(negative["absolute_gap"], 0.05)
        self.assertAlmostEqual(negative["mean_absolute_gap_across_subsets"], (0.05 + 6 * 0.02) / 7)

    def test_latex_table_has_all_regimes_and_no_missing_cells(self) -> None:
        aggregate = _aggregate_rows(self.seed_rows, list(DEFAULT_DATASETS))
        table = _latex_table(aggregate, list(DEFAULT_DATASETS), "macro_f1", "Caption", "tab:test")
        self.assertNotIn("N/A", table)
        for label in REGIME_LABELS.values():
            self.assertIn(label, table)
        self.assertIn("C+S+P", table)


if __name__ == "__main__":
    unittest.main()
