from __future__ import annotations

import json
from pathlib import Path

import pytest

from clauseforge.evaluation.dataset import EvaluationDataError, load_evaluation_dataset
from clauseforge.evaluation.runner import main, run


def _write_data(path: Path, *, leak: bool = False) -> None:
    path.mkdir()
    contracts = [
        {"contract_id": "c1", "raw_text": "law text"},
        {"contract_id": "c2", "raw_text": "term text"},
        {"contract_id": "c3", "raw_text": "law clause"},
    ]
    clauses = [
        {"clause_id": "a", "contract_id": "c1", "text": "law text", "category": "Law"},
        {
            "clause_id": "b",
            "contract_id": "c2",
            "text": "term text",
            "category": "Term",
        },
        {
            "clause_id": "c",
            "contract_id": "c3",
            "text": "law clause",
            "category": "Law",
        },
    ]
    for name, values in (("contracts.jsonl", contracts), ("clauses.jsonl", clauses)):
        (path / name).write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
        )
    splits = {
        "train": ["c1", "c2"],
        "validation": ["c3"],
        "test": ["c1"] if leak else [],
    }
    if not leak:
        splits["test"] = []
    (path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    for name in ("manifest.json",):
        (path / name).write_text("{}", encoding="utf-8")


def test_dataset_view_and_split_leakage_protection(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_data(data)
    dataset = load_evaluation_dataset(data)
    assert len(dataset.split("train")) == 2

    leaked = tmp_path / "leaked"
    _write_data(leaked, leak=True)
    with pytest.raises(EvaluationDataError, match="leakage"):
        load_evaluation_dataset(leaked)


def test_jsonl_reader_preserves_unicode_line_separator(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_data(data)
    contracts_path = data / "contracts.jsonl"
    content = contracts_path.read_text(encoding="utf-8")
    contracts_path.write_text(
        content.replace("law text", "law\u2028text", 1),
        encoding="utf-8",
        newline="\n",
    )

    dataset = load_evaluation_dataset(data)

    assert dataset.split("train")[0].contract_length == 8


def test_runner_serializes_results_and_cli_fails_for_missing_data(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    _write_data(data)

    summaries = run(
        data,
        output,
        models=["majority"],
        split="validation",
        bootstrap_iterations=5,
    )

    assert summaries[0]["model"] == "majority"
    assert (output / "majority" / "validation" / "metrics.json").is_file()
    assert (output / "RESULTS_VALIDATION.md").is_file()
    assert main(["--data", str(tmp_path / "missing"), "--output", str(output)]) == 1
