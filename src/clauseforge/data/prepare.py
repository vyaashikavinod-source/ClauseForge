"""End-to-end CUAD data preparation command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from clauseforge.data.cuad import CuadFormatError, load_cuad
from clauseforge.data.manifest import build_manifest, sha256_file
from clauseforge.data.models import (
    ClauseRecord,
    ContractRecord,
    JsonObject,
    SegmentRecord,
)
from clauseforge.data.normalize import (
    AnnotationAlignmentError,
    normalize_cuad,
    stable_id,
)
from clauseforge.data.segment import segment_contract
from clauseforge.data.split import SplitMap, split_contracts
from clauseforge.data.statistics import build_statistics
from clauseforge.data.validate import DatasetValidationError, validate_dataset

PROCESSING_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class PreparationResult:
    contracts: list[ContractRecord]
    clauses: list[ClauseRecord]
    segments: list[SegmentRecord]
    splits: SplitMap
    statistics: JsonObject
    output_files: tuple[Path, ...]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_jsonl(path: Path, records: Iterable[JsonObject]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            target.write("\n")


def prepare_cuad(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    dry_run: bool = False,
) -> PreparationResult:
    dataset = load_cuad(input_path)
    contracts, clauses = normalize_cuad(dataset, PROCESSING_VERSION)
    contracts.sort(key=lambda item: item.contract_id)
    clauses.sort(key=lambda item: (item.contract_id, item.start_char, item.clause_id))
    segments = [
        SegmentRecord(
            segment_id=stable_id(
                "segment",
                contract.contract_id,
                str(segment.start_char),
                str(segment.end_char),
                segment.strategy,
            ),
            contract_id=contract.contract_id,
            text=segment.text,
            start_char=segment.start_char,
            end_char=segment.end_char,
            strategy=segment.strategy,
        )
        for contract in contracts
        for segment in segment_contract(contract.raw_text)
    ]
    validation = validate_dataset(contracts, clauses)
    validation.raise_for_errors()
    splits = split_contracts(
        contracts,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    statistics = build_statistics(contracts, clauses, splits, validation)
    statistics["candidate_segment_count"] = len(segments)
    statistics["candidate_segment_strategies"] = dict(
        sorted(
            {
                strategy: sum(item.strategy == strategy for item in segments)
                for strategy in {item.strategy for item in segments}
            }.items()
        )
    )
    if dry_run:
        return PreparationResult(contracts, clauses, segments, splits, statistics, ())

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_path = output_dir / "contracts.jsonl"
    clauses_path = output_dir / "clauses.jsonl"
    segments_path = output_dir / "segments.jsonl"
    splits_path = output_dir / "splits.json"
    statistics_path = output_dir / "statistics.json"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(contracts_path, (item.to_dict() for item in contracts))
    _write_jsonl(clauses_path, (item.to_dict() for item in clauses))
    _write_jsonl(segments_path, (item.to_dict() for item in segments))
    splits_path.write_text(_json_text(splits), encoding="utf-8", newline="\n")
    statistics_path.write_text(_json_text(statistics), encoding="utf-8", newline="\n")
    deterministic_outputs = (
        contracts_path,
        clauses_path,
        segments_path,
        splits_path,
        statistics_path,
    )
    output_checksums = {path.name: sha256_file(path) for path in deterministic_outputs}
    configuration: JsonObject = {
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "split_unit": "contract",
        "annotation_policy": "authoritative_cuad_spans",
    }
    manifest = build_manifest(
        source_path=input_path,
        dataset_version=dataset.version,
        processing_version=PROCESSING_VERSION,
        contracts=contracts,
        clauses=clauses,
        splits=splits,
        configuration=configuration,
        output_checksums=output_checksums,
    )
    manifest_path.write_text(
        _json_text(manifest.to_dict()), encoding="utf-8", newline="\n"
    )
    return PreparationResult(
        contracts,
        clauses,
        segments,
        splits,
        statistics,
        (*deterministic_outputs, manifest_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local CUAD v1 data")
    parser.add_argument("--input", required=True, type=Path, help="CUAD_v1.json path")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_cuad(
            args.input,
            args.output,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            dry_run=args.dry_run,
        )
    except (
        AnnotationAlignmentError,
        CuadFormatError,
        DatasetValidationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"CUAD preparation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Prepared {len(result.contracts)} contracts and {len(result.clauses)} clauses"
    )
    if args.dry_run:
        print("Dry run: no output files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
