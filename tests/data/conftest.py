"""Shared test record factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from clauseforge.data.models import ContractRecord, Provenance


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "cuad_minimal.json"


def make_contract(index: int) -> ContractRecord:
    return ContractRecord(
        contract_id=f"contract-{index:02d}",
        source="CUAD",
        title=f"contract-{index:02d}.txt",
        raw_text="Contract text.",
        metadata={},
        provenance=Provenance("CUAD", "1.0", "1.0.0", "fixture", str(index)),
    )
