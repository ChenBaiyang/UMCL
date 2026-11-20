from __future__ import annotations

import inspect
import unittest
from collections import Counter

import torch

from models.fusion.evaluation import run_evaluation
from models.fusion.posteriors import product_of_experts
from models.fusion.schemas import FusionBatch
from models.fusion.subset_sampling import (
    CORE_VIEWS,
    SUBSET_SPECS,
    DeterministicMixedSubsetMasker,
)
from models.fusion.training import run_training


def _batch(event_ids: list[str], availability: list[tuple[int, int, int]]) -> FusionBatch:
    masks = {
        view: torch.tensor([row[index] for row in availability], dtype=torch.float32)
        for index, view in enumerate(CORE_VIEWS)
    }
    masks.update(
        {
            "evidence": torch.zeros(len(event_ids), dtype=torch.float32),
            "source": torch.zeros(len(event_ids), dtype=torch.float32),
        }
    )
    return FusionBatch(
        event_ids=event_ids,
        dataset_ids=["synthetic"] * len(event_ids),
        platforms=["test"] * len(event_ids),
        canonical_splits=["train"] * len(event_ids),
        raw_split_sources=[None] * len(event_ids),
        modalities={},
        modality_mask=masks,
        task_labels={"primary_task": torch.zeros(len(event_ids), dtype=torch.long)},
        task_mask={"primary_task": torch.ones(len(event_ids), dtype=torch.float32)},
        metadata=[{} for _ in event_ids],
    )


class DeterministicMixedSubsetMaskerTests(unittest.TestCase):
    def test_full_availability_is_approximately_uniform(self) -> None:
        masker = DeterministicMixedSubsetMasker(seed=2, phase="train", vary_by_epoch=True)
        count = 14000
        observed = Counter(
            masker.choose(
                f"event-{index}",
                {"content": True, "stance": True, "propagation": True},
                epoch=3,
            ).requested_subset
            for index in range(count)
        )
        expected = count / len(SUBSET_SPECS)
        self.assertEqual(set(observed), {name for name, _ in SUBSET_SPECS})
        self.assertLess(max(abs(value - expected) for value in observed.values()), expected * 0.08)

    def test_same_seed_and_input_are_reproducible(self) -> None:
        left = DeterministicMixedSubsetMasker(seed=16, phase="train", vary_by_epoch=True)
        right = DeterministicMixedSubsetMasker(seed=16, phase="train", vary_by_epoch=True)
        availability = {view: True for view in CORE_VIEWS}
        self.assertEqual(
            left.choose("event-7", availability, epoch=5),
            right.choose("event-7", availability, epoch=5),
        )

    def test_training_assignments_change_across_epochs(self) -> None:
        masker = DeterministicMixedSubsetMasker(seed=8, phase="train", vary_by_epoch=True)
        availability = {view: True for view in CORE_VIEWS}
        changed = sum(
            masker.choose(f"event-{index}", availability, epoch=1).requested_subset
            != masker.choose(f"event-{index}", availability, epoch=2).requested_subset
            for index in range(100)
        )
        self.assertGreater(changed, 50)

    def test_validation_assignments_are_fixed_across_epochs(self) -> None:
        masker = DeterministicMixedSubsetMasker(seed=4, phase="validation", vary_by_epoch=False)
        availability = {view: True for view in CORE_VIEWS}
        self.assertEqual(
            masker.choose("event-11", availability, epoch=1).requested_subset,
            masker.choose("event-11", availability, epoch=29).requested_subset,
        )

    def test_intrinsic_missing_stance_never_produces_empty_observation(self) -> None:
        masker = DeterministicMixedSubsetMasker(seed=32, phase="train", vary_by_epoch=True)
        availability = {"content": True, "stance": False, "propagation": True}
        resampled = []
        for index in range(1000):
            assignment = masker.choose(f"pheme-{index}", availability, epoch=1)
            self.assertTrue(assignment.observed_views)
            self.assertNotIn("stance", assignment.observed_views)
            if assignment.resampled_empty_request:
                resampled.append(assignment)
        self.assertTrue(resampled)
        self.assertTrue(all(item.requested_subset != "S" for item in resampled))

    def test_apply_intersects_sampled_and_intrinsic_masks(self) -> None:
        batch = _batch(["a", "b", "c"], [(1, 1, 1), (1, 0, 1), (0, 1, 0)])
        masker = DeterministicMixedSubsetMasker(seed=2, phase="train", vary_by_epoch=True)
        assignments = masker.apply(batch, epoch=1)
        observed_counts = sum(batch.modality_mask[view] for view in CORE_VIEWS)
        self.assertTrue(torch.all(observed_counts > 0))
        self.assertEqual(assignments[2].observed_views, ("stance",))
        self.assertEqual(float(batch.modality_mask["content"][2]), 0.0)
        self.assertEqual(float(batch.modality_mask["propagation"][2]), 0.0)

    def test_apply_masks_stance_auxiliary_labels_for_unobserved_events(self) -> None:
        batch = _batch(["a", "b"], [(1, 1, 1), (1, 1, 1)])
        batch.modalities["stance"] = {
            "event_index": torch.tensor([0, 0, 1], dtype=torch.long),
            "reply_label_mask": torch.ones(3, dtype=torch.float32),
        }
        masker = DeterministicMixedSubsetMasker(seed=2, phase="train", vary_by_epoch=True)
        masker.apply(batch, epoch=1)
        expected = batch.modality_mask["stance"][batch.modalities["stance"]["event_index"]]
        self.assertTrue(torch.equal(batch.modalities["stance"]["reply_label_mask"], expected))

    def test_zero_mask_expert_contributes_no_precision(self) -> None:
        mu_a = torch.tensor([[0.5, -0.25]])
        logvar_a = torch.tensor([[0.1, -0.2]])
        mu_b = torch.tensor([[100.0, -100.0]])
        logvar_b = torch.tensor([[-6.0, -6.0]])
        one = torch.ones(1)
        zero = torch.zeros(1)
        reference = product_of_experts([mu_a], [logvar_a], [one])
        masked = product_of_experts([mu_a, mu_b], [logvar_a, logvar_b], [one, zero])
        self.assertTrue(torch.allclose(reference[0], masked[0]))
        self.assertTrue(torch.allclose(reference[1], masked[1]))

    def test_existing_training_and_evaluation_defaults_remain_disabled(self) -> None:
        training_signature = inspect.signature(run_training)
        evaluation_signature = inspect.signature(run_evaluation)
        self.assertIsNone(training_signature.parameters["train_subset_masker"].default)
        self.assertIsNone(training_signature.parameters["eval_subset_masker"].default)
        self.assertIsNone(evaluation_signature.parameters["subset_masker"].default)


if __name__ == "__main__":
    unittest.main()
