"""Machine-readable dataset statistics."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median

from clauseforge.data.models import ClauseRecord, ContractRecord, JsonObject
from clauseforge.data.split import SplitMap
from clauseforge.data.validate import ValidationReport


def _distribution(lengths: list[int]) -> JsonObject:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 3),
        "median": round(median(lengths), 3),
    }


def build_statistics(
    contracts: list[ContractRecord],
    clauses: list[ClauseRecord],
    splits: SplitMap,
    validation: ValidationReport,
) -> JsonObject:
    counts = Counter(clause.category for clause in clauses)
    annotated_contracts = {clause.contract_id for clause in clauses}
    return {
        "contract_count": len(contracts),
        "clause_count": len(clauses),
        "category_count": len(counts),
        "counts_per_category": dict(sorted(counts.items())),
        "contract_text_length": _distribution(
            [len(contract.raw_text) for contract in contracts]
        ),
        "clause_text_length": _distribution([len(clause.text) for clause in clauses]),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "contracts_with_zero_annotations": sorted(
            contract.contract_id
            for contract in contracts
            if contract.contract_id not in annotated_contracts
        ),
        "validation_failure_count": len(validation.issues),
        "overlapping_annotation_pair_count": validation.overlap_count,
    }
