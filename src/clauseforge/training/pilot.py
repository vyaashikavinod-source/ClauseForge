"""Deterministic bounded Phase 3B pilot dataset selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import cast

from clauseforge.training.dataset import TrainingDataset
from clauseforge.training.templates import TrainingExample

PILOT_LABEL = "PHASE 3B T4 PILOT — NOT FINAL MODEL PERFORMANCE"


def selection_checksum(example_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(example_ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PilotSelection:
    split: str
    requested_count: int
    selected: tuple[TrainingExample, ...]
    category_counts: dict[str, int]
    supported_category_count: int
    covered_category_count: int
    selected_example_ids: tuple[str, ...]
    checksum: str

    def metadata(self) -> dict[str, object]:
        return {
            "label": PILOT_LABEL,
            "split": self.split,
            "requested_count": self.requested_count,
            "selected_count": len(self.selected),
            "supported_category_count": self.supported_category_count,
            "covered_category_count": self.covered_category_count,
            "category_counts": self.category_counts,
            "selected_example_ids": list(self.selected_example_ids),
            "checksum": self.checksum,
            "test_examples_selected": 0,
        }


def stratified_selection(
    examples: tuple[TrainingExample, ...], *, count: int, seed: int, split: str
) -> PilotSelection:
    """Round-robin categories after seeded within-category shuffling."""
    if count <= 0:
        raise ValueError("pilot sample count must be positive")
    if count > len(examples):
        raise ValueError(f"pilot {split} count exceeds available examples")
    if any(example.split != split for example in examples):
        raise ValueError(f"pilot {split} selection received another split")
    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[example.target_label].append(example)
    generator = random.Random(seed)
    for values in groups.values():
        generator.shuffle(values)
    labels = sorted(groups)
    selected: list[TrainingExample] = []
    position = 0
    while len(selected) < count:
        progressed = False
        for label in labels:
            values = groups[label]
            if position < len(values) and len(selected) < count:
                selected.append(values[position])
                progressed = True
        if not progressed:
            break
        position += 1
    ids = tuple(example.clause_id for example in selected)
    checksum = selection_checksum(ids)
    counts = Counter(example.target_label for example in selected)
    return PilotSelection(
        split=split,
        requested_count=count,
        selected=tuple(selected),
        category_counts=dict(sorted(counts.items())),
        supported_category_count=len(groups),
        covered_category_count=len(counts),
        selected_example_ids=ids,
        checksum=checksum,
    )


def build_pilot_dataset(
    source: TrainingDataset,
    *,
    train_count: int = 512,
    validation_count: int = 256,
    seed: int = 42,
) -> tuple[TrainingDataset, PilotSelection, PilotSelection]:
    train = stratified_selection(
        source.train, count=train_count, seed=seed, split="train"
    )
    validation = stratified_selection(
        source.validation, count=validation_count, seed=seed, split="validation"
    )
    manifest = {
        **source.manifest,
        "label": PILOT_LABEL,
        "pilot": True,
        "pilot_seed": seed,
        "pilot_train": train.metadata(),
        "pilot_validation": validation.metadata(),
        "source_training_example_count": len(source.train),
        "source_validation_example_count": len(source.validation),
        "training_example_count": len(train.selected),
        "validation_example_count": len(validation.selected),
        "test_split_access": "integrity check only; zero test examples selected",
    }
    return (
        TrainingDataset(train.selected, validation.selected, source.taxonomy, manifest),
        train,
        validation,
    )


def combined_selection_checksum(
    train: PilotSelection, validation: PilotSelection
) -> str:
    return hashlib.sha256(
        f"{train.checksum}:{validation.checksum}".encode()
    ).hexdigest()


def restore_pilot_selection(
    examples: tuple[TrainingExample, ...], metadata: dict[str, object], *, split: str
) -> PilotSelection:
    """Restore and validate an ordered selection from persisted example IDs."""
    if metadata.get("split") != split:
        raise ValueError(f"persisted {split} selection has the wrong split")
    raw_ids = metadata.get("selected_example_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(item, str) for item in raw_ids
    ):
        raise ValueError(f"persisted {split} selection IDs are malformed")
    ids = tuple(raw_ids)
    if len(ids) != len(set(ids)):
        raise ValueError(f"persisted {split} selection contains duplicate IDs")
    by_id = {item.clause_id: item for item in examples if item.split == split}
    if len(by_id) != len(examples) or any(item not in by_id for item in ids):
        raise ValueError(f"persisted {split} selection contains unknown IDs")
    selected = tuple(by_id[item] for item in ids)
    checksum = selection_checksum(ids)
    requested = metadata.get("requested_count")
    selected_count = metadata.get("selected_count")
    stored_checksum = metadata.get("checksum")
    counts = dict(sorted(Counter(item.target_label for item in selected).items()))
    stored_counts = metadata.get("category_counts")
    if stored_checksum != checksum:
        raise ValueError(
            f"persisted {split} selection checksum differs: "
            f"stored={stored_checksum}, calculated={checksum}"
        )
    if (
        not isinstance(requested, int)
        or requested != len(ids)
        or selected_count != len(ids)
        or stored_counts != counts
    ):
        raise ValueError(f"persisted {split} selection metadata is inconsistent")
    supported = len({item.target_label for item in examples})
    covered = len(counts)
    if (
        metadata.get("supported_category_count") != supported
        or metadata.get("covered_category_count") != covered
        or metadata.get("test_examples_selected") != 0
    ):
        raise ValueError(f"persisted {split} selection coverage is inconsistent")
    return PilotSelection(
        split=split,
        requested_count=requested,
        selected=selected,
        category_counts=cast(dict[str, int], stored_counts),
        supported_category_count=supported,
        covered_category_count=covered,
        selected_example_ids=ids,
        checksum=checksum,
    )


def diagnostic_validation_selection(
    examples: tuple[TrainingExample, ...],
    historical: PilotSelection,
    *,
    requested_count: int | None,
    seed: int,
) -> tuple[PilotSelection, bool]:
    """Use historical validation IDs unless an explicit new size is requested."""
    count = historical.requested_count if requested_count is None else requested_count
    if count == historical.requested_count:
        return historical, False
    return (
        stratified_selection(examples, count=count, seed=seed, split="validation"),
        True,
    )
