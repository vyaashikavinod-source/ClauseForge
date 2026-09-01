"""Dataset invariant validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from clauseforge.data.models import ClauseRecord, ContractRecord


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    overlap_count: int

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.issues[:10]
            )
            raise DatasetValidationError(details)


class DatasetValidationError(ValueError):
    """Raised when normalized records violate dataset invariants."""


def validate_dataset(
    contracts: list[ContractRecord], clauses: list[ClauseRecord]
) -> ValidationReport:
    """Validate identifiers, ownership, offsets, spans, and ordering invariants."""
    issues: list[ValidationIssue] = []
    contracts_by_id: dict[str, ContractRecord] = {}
    for contract in contracts:
        if contract.contract_id in contracts_by_id:
            issues.append(
                ValidationIssue(
                    "duplicate_contract_id",
                    f"contract ID {contract.contract_id!r} occurs more than once",
                    contract.contract_id,
                )
            )
        contracts_by_id[contract.contract_id] = contract
        if not contract.raw_text.strip():
            issues.append(
                ValidationIssue(
                    "empty_contract", "contract text is empty", contract.contract_id
                )
            )

    seen_clause_ids: set[str] = set()
    spans_by_contract: dict[str, list[tuple[int, int]]] = {}
    for clause in clauses:
        if clause.clause_id in seen_clause_ids:
            issues.append(
                ValidationIssue(
                    "duplicate_clause_id",
                    f"clause ID {clause.clause_id!r} occurs more than once",
                    clause.clause_id,
                )
            )
        seen_clause_ids.add(clause.clause_id)
        owner = contracts_by_id.get(clause.contract_id)
        if owner is None:
            issues.append(
                ValidationIssue(
                    "missing_contract",
                    f"contract {clause.contract_id!r} does not exist",
                    clause.clause_id,
                )
            )
            continue
        if not clause.text:
            issues.append(
                ValidationIssue(
                    "empty_clause", "clause text is empty", clause.clause_id
                )
            )
        if not (0 <= clause.start_char < clause.end_char <= len(owner.raw_text)):
            issues.append(
                ValidationIssue(
                    "invalid_offsets",
                    f"span [{clause.start_char}:{clause.end_char}] is outside contract",
                    clause.clause_id,
                )
            )
        elif owner.raw_text[clause.start_char : clause.end_char] != clause.text:
            issues.append(
                ValidationIssue(
                    "span_mismatch",
                    "clause text does not equal the contract text at its offsets",
                    clause.clause_id,
                )
            )
        spans_by_contract.setdefault(clause.contract_id, []).append(
            (clause.start_char, clause.end_char)
        )

    overlap_count = 0
    for spans in spans_by_contract.values():
        ordered = sorted(spans)
        for (_, previous_end), (current_start, _) in pairwise(ordered):
            if current_start < previous_end:
                overlap_count += 1
    return ValidationReport(tuple(issues), overlap_count)
