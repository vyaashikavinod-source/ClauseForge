from pathlib import Path

import pytest

from clauseforge.data.cuad import load_cuad
from clauseforge.data.normalize import (
    AnnotationAlignmentError,
    align_annotation,
    normalize_cuad,
)


def test_normalization_preserves_annotations_and_offsets(fixture_path: Path) -> None:
    contracts, clauses = normalize_cuad(load_cuad(fixture_path), "1.0.0")

    assert len(contracts) == 2
    assert [clause.category for clause in clauses] == [
        "Effective Date",
        "Governing Law",
    ]
    for clause in clauses:
        contract = next(
            item for item in contracts if item.contract_id == clause.contract_id
        )
        assert contract.raw_text[clause.start_char : clause.end_char] == clause.text
        assert clause.provenance.original_start is not None


def test_deterministic_identifiers(fixture_path: Path) -> None:
    first = normalize_cuad(load_cuad(fixture_path), "1.0.0")
    second = normalize_cuad(load_cuad(fixture_path), "1.0.0")

    assert [item.contract_id for item in first[0]] == [
        item.contract_id for item in second[0]
    ]
    assert [item.clause_id for item in first[1]] == [
        item.clause_id for item in second[1]
    ]


def test_repeated_text_uses_authoritative_offset() -> None:
    text = "Alpha then Alpha"

    span, end, alignment = align_annotation(text, "Alpha", 11)

    assert (span, end, alignment) == ("Alpha", 16, "exact")


def test_whitespace_normalization_is_reported_without_relocation() -> None:
    span, end, alignment = align_annotation("A \t \nB", "A B", 0)

    assert span == "A \t \nB"
    assert end == 6
    assert alignment == "whitespace_normalized"


@pytest.mark.parametrize("start", [-1, 20])
def test_out_of_range_offset_fails(start: int) -> None:
    with pytest.raises(AnnotationAlignmentError, match="outside context"):
        align_annotation("short text", "text", start)


def test_misalignment_is_not_silently_fixed() -> None:
    with pytest.raises(AnnotationAlignmentError, match="does not match"):
        align_annotation("Alpha then Alpha", "Alpha", 1)
