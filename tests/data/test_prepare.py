from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clauseforge.data.prepare import main, prepare_cuad


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_end_to_end_preparation_and_deterministic_outputs(
    fixture_path: Path, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_cuad(fixture_path, first_dir)
    second = prepare_cuad(fixture_path, second_dir)

    expected = {
        "contracts.jsonl",
        "clauses.jsonl",
        "segments.jsonl",
        "splits.json",
        "statistics.json",
    }
    for name in expected:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert {path.name for path in first.output_files} == expected | {"manifest.json"}
    contracts = _jsonl(first_dir / "contracts.jsonl")
    clauses = _jsonl(first_dir / "clauses.jsonl")
    segments = _jsonl(first_dir / "segments.jsonl")
    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((first_dir / "splits.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == len(contracts) == 2
    assert manifest["clause_count"] == len(clauses) == 2
    assigned = [contract_id for values in splits.values() for contract_id in values]
    assert len(assigned) == len(set(assigned)) == len(contracts)
    contracts_by_id = {item["contract_id"]: item for item in contracts}
    for clause in clauses:
        text = contracts_by_id[clause["contract_id"]]["raw_text"]
        assert text[clause["start_char"] : clause["end_char"]] == clause["text"]
    for segment in segments:
        text = contracts_by_id[segment["contract_id"]]["raw_text"]
        assert text[segment["start_char"] : segment["end_char"]] == segment["text"]
    assert first.statistics == second.statistics


def test_dry_run_and_cli_failure_status(fixture_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "dry"

    assert (
        main(["--input", str(fixture_path), "--output", str(output), "--dry-run"]) == 0
    )
    assert not output.exists()
    assert main(["--input", str(tmp_path / "missing"), "--output", str(output)]) == 1
