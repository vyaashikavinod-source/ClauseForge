import pytest

from clauseforge.data.split import split_contracts, validate_splits
from clauseforge.data.validate import DatasetValidationError
from tests.data.conftest import make_contract


def test_split_is_deterministic_contract_level_and_uses_default_ratios() -> None:
    contracts = [make_contract(index) for index in range(10)]

    first = split_contracts(contracts, seed=7)
    second = split_contracts(list(reversed(contracts)), seed=7)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assigned = [contract_id for values in first.values() for contract_id in values]
    assert len(assigned) == len(set(assigned)) == 10


def test_invalid_ratios_fail() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_contracts([make_contract(1)], train_ratio=0.5)


def test_leakage_is_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="more than one split"):
        validate_splits({"train": ["a"], "validation": ["a"], "test": []}, {"a"})
