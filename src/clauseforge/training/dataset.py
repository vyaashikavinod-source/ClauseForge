"""Leakage-safe conversion of processed CUAD clauses to SFT examples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from clauseforge.evaluation.dataset import load_evaluation_dataset
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION, TrainingExample


class TrainingDataError(ValueError):
    """Raised when training data violates taxonomy or split invariants."""


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    taxonomy: tuple[str, ...]
    manifest: dict[str, object]


def _checksum(examples: tuple[TrainingExample, ...]) -> str:
    payload = "\n".join(
        json.dumps(asdict(example), sort_keys=True, ensure_ascii=False)
        for example in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_training_dataset(
    data_dir: Path,
    *,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
) -> TrainingDataset:
    source = load_evaluation_dataset(data_dir)
    taxonomy = tuple(source.labels)
    if len(taxonomy) != 41:
        raise TrainingDataError(
            f"expected 41 authoritative categories, found {len(taxonomy)}"
        )

    def convert(split: str, limit: int | None) -> tuple[TrainingExample, ...]:
        values = source.split(split)
        if limit is not None:
            if limit <= 0:
                raise TrainingDataError("example limits must be positive")
            values = values[:limit]
        examples = tuple(
            TrainingExample(
                item.clause_id, item.contract_id, item.text, item.label, split
            )
            for item in values
        )
        unknown = {item.target_label for item in examples} - set(taxonomy)
        if unknown:
            raise TrainingDataError(f"unknown target categories: {sorted(unknown)}")
        return examples

    train = convert("train", max_train_examples)
    validation = convert("validation", max_validation_examples)
    train_contracts = {item.contract_id for item in train}
    validation_contracts = {item.contract_id for item in validation}
    test_contracts = {item.contract_id for item in source.split("test")}
    if train_contracts & validation_contracts or train_contracts & test_contracts:
        raise TrainingDataError("contract leakage into optimization data")
    if validation_contracts & test_contracts:
        raise TrainingDataError("contract leakage into model-selection data")
    split_checksums = {"train": _checksum(train), "validation": _checksum(validation)}
    dataset_checksum = hashlib.sha256(
        (split_checksums["train"] + split_checksums["validation"]).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "training_example_count": len(train),
        "validation_example_count": len(validation),
        "class_distribution": dict(
            sorted(Counter(x.target_label for x in train).items())
        ),
        "dataset_checksum": dataset_checksum,
        "split_checksums": split_checksums,
        "taxonomy": list(taxonomy),
        "test_split_access": (
            "integrity check only; no examples materialized for training"
        ),
    }
    return TrainingDataset(train, validation, taxonomy, manifest)
