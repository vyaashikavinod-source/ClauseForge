"""Leakage-safe evaluation views over processed ClauseForge data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class EvaluationDataError(ValueError):
    """Raised when processed data cannot support leakage-safe evaluation."""


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    clause_id: str
    contract_id: str
    text: str
    label: str
    clause_length: int
    contract_length: int


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    examples_by_split: dict[str, list[EvaluationExample]]
    labels: list[str]
    data_dir: Path

    def split(self, name: str) -> list[EvaluationExample]:
        if name not in self.examples_by_split:
            raise EvaluationDataError(f"unknown split: {name}")
        return self.examples_by_split[name]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise EvaluationDataError(f"required data file does not exist: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise EvaluationDataError(
                    f"record at {path}:{line_number} is not an object"
                )
            records.append(value)
    return records


def load_evaluation_dataset(data_dir: Path) -> EvaluationDataset:
    contracts = _read_jsonl(data_dir / "contracts.jsonl")
    clauses = _read_jsonl(data_dir / "clauses.jsonl")
    try:
        splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"unable to read splits.json: {exc}") from exc
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise EvaluationDataError(
            "splits.json must contain train, validation, and test"
        )

    contract_lengths: dict[str, int] = {}
    for record in contracts:
        contract_id, raw_text = record.get("contract_id"), record.get("raw_text")
        if not isinstance(contract_id, str) or not isinstance(raw_text, str):
            raise EvaluationDataError(
                "contract records require string IDs and raw_text"
            )
        contract_lengths[contract_id] = len(raw_text)

    split_by_contract: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        values = splits[split_name]
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise EvaluationDataError(f"split {split_name} must contain contract IDs")
        for contract_id in values:
            if contract_id in split_by_contract:
                raise EvaluationDataError(f"contract leakage detected: {contract_id}")
            split_by_contract[contract_id] = split_name
    if set(split_by_contract) != set(contract_lengths):
        raise EvaluationDataError("split assignments do not exactly cover contracts")

    examples_by_split: dict[str, list[EvaluationExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    labels: set[str] = set()
    for record in clauses:
        fields = (
            record.get("clause_id"),
            record.get("contract_id"),
            record.get("text"),
            record.get("category"),
        )
        if not all(isinstance(item, str) for item in fields):
            raise EvaluationDataError(
                "clause records require string IDs, text, and category"
            )
        clause_id, contract_id, text, label = fields
        assert isinstance(clause_id, str)
        assert isinstance(contract_id, str)
        assert isinstance(text, str)
        assert isinstance(label, str)
        owner_split = split_by_contract.get(contract_id)
        if owner_split is None:
            raise EvaluationDataError(f"clause contract has no split: {contract_id}")
        labels.add(label)
        examples_by_split[owner_split].append(
            EvaluationExample(
                clause_id,
                contract_id,
                text,
                label,
                len(text),
                contract_lengths[contract_id],
            )
        )
    for examples in examples_by_split.values():
        examples.sort(key=lambda item: item.clause_id)
    return EvaluationDataset(examples_by_split, sorted(labels), data_dir)
