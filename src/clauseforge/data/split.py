"""Deterministic contract-level dataset splitting."""

from __future__ import annotations

import hashlib
import math

from clauseforge.data.models import ContractRecord
from clauseforge.data.validate import DatasetValidationError

SplitMap = dict[str, list[str]]


def _allocation(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [count * ratio for ratio in ratios]
    allocated = [math.floor(value) for value in exact]
    remainder = count - sum(allocated)
    priority = sorted(
        range(3),
        key=lambda index: (exact[index] - allocated[index], -index),
        reverse=True,
    )
    for index in priority[:remainder]:
        allocated[index] += 1
    return allocated[0], allocated[1], allocated[2]


def split_contracts(
    contracts: list[ContractRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitMap:
    """Assign each unique contract to exactly one stable split."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must be non-negative and sum to 1.0")
    contract_ids = [contract.contract_id for contract in contracts]
    if len(contract_ids) != len(set(contract_ids)):
        raise DatasetValidationError("cannot split duplicate contract IDs")

    def order_key(contract_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{contract_id}".encode()).hexdigest()
        return digest, contract_id

    ordered = sorted(contract_ids, key=order_key)
    train_count, validation_count, _ = _allocation(len(ordered), ratios)
    train_end = train_count
    validation_end = train_end + validation_count
    splits = {
        "train": sorted(ordered[:train_end]),
        "validation": sorted(ordered[train_end:validation_end]),
        "test": sorted(ordered[validation_end:]),
    }
    validate_splits(splits, set(contract_ids))
    return splits


def validate_splits(splits: SplitMap, expected_contract_ids: set[str]) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise DatasetValidationError("splits must contain train, validation, and test")
    assigned: list[str] = [item for values in splits.values() for item in values]
    if len(assigned) != len(set(assigned)):
        raise DatasetValidationError("a contract appears in more than one split")
    if set(assigned) != expected_contract_ids:
        missing = expected_contract_ids - set(assigned)
        extra = set(assigned) - expected_contract_ids
        raise DatasetValidationError(
            "split membership mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
