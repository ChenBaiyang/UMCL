from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from models.fusion.schemas import FusionBatch


CORE_VIEWS: tuple[str, ...] = ("content", "stance", "propagation")
SUBSET_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("C", ("content",)),
    ("S", ("stance",)),
    ("P", ("propagation",)),
    ("C+S", ("content", "stance")),
    ("C+P", ("content", "propagation")),
    ("S+P", ("stance", "propagation")),
    ("C+S+P", ("content", "stance", "propagation")),
)
SAMPLING_ALGORITHM_VERSION = "sha256-ranked-subsets-v1"


@dataclass(frozen=True)
class SubsetAssignment:
    event_id: str
    phase: str
    epoch_key: int
    requested_subset: str
    observed_subset: str
    requested_views: tuple[str, ...]
    observed_views: tuple[str, ...]
    intrinsically_available_views: tuple[str, ...]
    resampled_empty_request: bool


class DeterministicMixedSubsetMasker:
    """Apply reproducible, event-level C/S/P availability masks to a fusion batch.

    Candidate subsets receive independent SHA-256 scores and are ranked by score.
    The first candidate whose intersection with the event's intrinsic availability
    is non-empty is selected. With all three views available, the first-ranked
    candidate is therefore uniform over the seven non-empty subsets. Ranking also
    provides a deterministic, finite fallback for intrinsically missing views.
    """

    def __init__(
        self,
        *,
        seed: int,
        phase: str,
        vary_by_epoch: bool,
        record_assignments: bool = False,
    ) -> None:
        normalized_phase = str(phase).strip().lower()
        if not normalized_phase:
            raise ValueError("phase must be non-empty")
        self.seed = int(seed)
        self.phase = normalized_phase
        self.vary_by_epoch = bool(vary_by_epoch)
        self.record_assignments = bool(record_assignments)
        self._recorded: dict[tuple[int, str], SubsetAssignment] = {}

    def _epoch_key(self, epoch: int) -> int:
        return int(epoch) if self.vary_by_epoch else 0

    def _score(self, event_id: str, epoch_key: int, subset_name: str) -> bytes:
        material = "|".join(
            (
                SAMPLING_ALGORITHM_VERSION,
                str(self.seed),
                self.phase,
                str(epoch_key),
                str(event_id),
                subset_name,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).digest()

    def choose(
        self,
        event_id: str,
        availability: Mapping[str, bool | int | float],
        *,
        epoch: int,
    ) -> SubsetAssignment:
        available = tuple(view for view in CORE_VIEWS if bool(availability.get(view, False)))
        if not available:
            raise ValueError(f"Event {event_id!r} has no intrinsically available core view.")
        available_set = set(available)
        epoch_key = self._epoch_key(epoch)
        ranked = sorted(
            SUBSET_SPECS,
            key=lambda spec: (self._score(str(event_id), epoch_key, spec[0]), spec[0]),
        )
        first_name, first_views = ranked[0]
        selected_name: str | None = None
        selected_views: tuple[str, ...] | None = None
        observed_views: tuple[str, ...] | None = None
        for name, requested_views in ranked:
            intersection = tuple(view for view in requested_views if view in available_set)
            if intersection:
                selected_name = name
                selected_views = requested_views
                observed_views = intersection
                break
        if selected_name is None or selected_views is None or observed_views is None:
            raise RuntimeError(f"Unable to choose a non-empty subset for event {event_id!r}.")
        observed_name = "+".join({"content": "C", "stance": "S", "propagation": "P"}[view] for view in observed_views)
        assignment = SubsetAssignment(
            event_id=str(event_id),
            phase=self.phase,
            epoch_key=epoch_key,
            requested_subset=selected_name,
            observed_subset=observed_name,
            requested_views=tuple(selected_views),
            observed_views=tuple(observed_views),
            intrinsically_available_views=available,
            resampled_empty_request=not any(view in available_set for view in first_views),
        )
        if self.record_assignments:
            self._recorded[(epoch_key, str(event_id))] = assignment
        return assignment

    def apply(self, batch: FusionBatch, *, epoch: int) -> list[SubsetAssignment]:
        missing_mask_keys = [view for view in CORE_VIEWS if view not in batch.modality_mask]
        if missing_mask_keys:
            raise KeyError(f"Fusion batch lacks core modality masks: {missing_mask_keys}")
        batch_size = len(batch.event_ids)
        for view in CORE_VIEWS:
            mask = batch.modality_mask[view]
            if mask.ndim != 1 or int(mask.shape[0]) != batch_size:
                raise ValueError(
                    f"Expected modality_mask[{view!r}] shape [{batch_size}], got {tuple(mask.shape)}"
                )

        assignments: list[SubsetAssignment] = []
        sampled_values = {view: [0.0] * batch_size for view in CORE_VIEWS}
        intrinsic_values = {
            view: batch.modality_mask[view].detach().cpu().tolist() for view in CORE_VIEWS
        }
        for row_index, event_id in enumerate(batch.event_ids):
            availability = {
                view: bool(intrinsic_values[view][row_index] > 0)
                for view in CORE_VIEWS
            }
            assignment = self.choose(str(event_id), availability, epoch=epoch)
            assignments.append(assignment)
            for view in assignment.observed_views:
                sampled_values[view][row_index] = 1.0

        for view in CORE_VIEWS:
            intrinsic_mask = batch.modality_mask[view]
            sampled_mask = torch.tensor(
                sampled_values[view], dtype=intrinsic_mask.dtype, device=intrinsic_mask.device
            )
            batch.modality_mask[view] = intrinsic_mask * sampled_mask
        stance_payload = batch.modalities.get("stance")
        if stance_payload is not None:
            reply_event_index = stance_payload.get("event_index")
            reply_label_mask = stance_payload.get("reply_label_mask")
            if isinstance(reply_event_index, torch.Tensor) and isinstance(reply_label_mask, torch.Tensor):
                if reply_event_index.numel() != reply_label_mask.numel():
                    raise ValueError("Stance reply event indices and label mask have different lengths.")
                if reply_event_index.numel() > 0:
                    reply_observed = batch.modality_mask["stance"][reply_event_index.long()]
                    stance_payload["reply_label_mask"] = reply_label_mask * reply_observed.to(
                        dtype=reply_label_mask.dtype
                    )
        observed_count = sum(batch.modality_mask[view] > 0 for view in CORE_VIEWS)
        if not bool(torch.all(observed_count > 0).item()):
            raise AssertionError("Mixed-subset masking produced an event with no observed core view.")
        return assignments

    def manifest(self) -> dict[str, Any]:
        return {
            "algorithm_version": SAMPLING_ALGORITHM_VERSION,
            "seed": self.seed,
            "phase": self.phase,
            "vary_by_epoch": self.vary_by_epoch,
            "epoch_key_rule": "actual_epoch" if self.vary_by_epoch else "constant_zero",
            "candidate_subsets": [
                {"name": name, "views": list(views)} for name, views in SUBSET_SPECS
            ],
            "selection_rule": (
                "Rank candidates by SHA-256(version|seed|phase|epoch_key|event_id|subset); "
                "choose the first candidate with a non-empty intersection with intrinsic availability."
            ),
        }

    def recorded_assignment_rows(self) -> list[dict[str, Any]]:
        return [
            asdict(assignment)
            for _, assignment in sorted(self._recorded.items(), key=lambda item: item[0])
        ]
