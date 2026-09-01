from clauseforge.data.models import ClauseRecord, ContractRecord, Provenance
from clauseforge.data.validate import validate_dataset


def _provenance() -> Provenance:
    return Provenance("CUAD", "1.0", "1.0.0", "fixture", "contract")


def test_valid_and_overlapping_annotations_are_preserved() -> None:
    contract = ContractRecord("c", "CUAD", "c.txt", "abcdef", {}, _provenance())
    clauses = [
        ClauseRecord("a", "c", "A", "abcd", 0, 4, "CUAD", _provenance()),
        ClauseRecord("b", "c", "B", "cdef", 2, 6, "CUAD", _provenance()),
    ]

    report = validate_dataset([contract], clauses)

    assert report.is_valid
    assert report.overlap_count == 1


def test_invalid_offsets_and_span_text_are_reported() -> None:
    contract = ContractRecord("c", "CUAD", "c.txt", "abcdef", {}, _provenance())
    clauses = [
        ClauseRecord("a", "c", "A", "wrong", 0, 5, "CUAD", _provenance()),
        ClauseRecord("b", "c", "B", "x", 9, 10, "CUAD", _provenance()),
    ]

    codes = {issue.code for issue in validate_dataset([contract], clauses).issues}

    assert codes == {"span_mismatch", "invalid_offsets"}
