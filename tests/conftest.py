"""Shared deterministic fixtures that never require ignored CUAD data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clauseforge.taxonomy import load_taxonomy_metadata


@pytest.fixture(scope="session")
def processed_cuad_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a synthetic, split-safe 41-category processed-data fixture."""
    output = tmp_path_factory.mktemp("processed-cuad-fixture")
    contracts: list[dict[str, str]] = []
    clauses: list[dict[str, str]] = []
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    counts = {"train": 13, "validation": 7, "test": 1}
    for category_index, category in enumerate(load_taxonomy_metadata()):
        for split, count in counts.items():
            for example_index in range(count):
                contract_id = (
                    f"{split}-contract-{category_index:02d}-{example_index:02d}"
                )
                clause_id = f"{split}-clause-{category_index:02d}-{example_index:02d}"
                text = f"Synthetic {category.category_name} clause {example_index}."
                contracts.append({"contract_id": contract_id, "raw_text": text})
                clauses.append(
                    {
                        "clause_id": clause_id,
                        "contract_id": contract_id,
                        "text": text,
                        "category": category.canonical,
                    }
                )
                splits[split].append(contract_id)
    for name, rows in (("contracts.jsonl", contracts), ("clauses.jsonl", clauses)):
        (output / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output / "splits.json").write_text(
        json.dumps(splits, sort_keys=True), encoding="utf-8"
    )
    return output
