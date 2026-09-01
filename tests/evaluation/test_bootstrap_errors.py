from clauseforge.evaluation.bootstrap import bootstrap_confidence_intervals
from clauseforge.evaluation.dataset import EvaluationExample
from clauseforge.evaluation.errors import build_errors, summarize_errors


def test_contract_bootstrap_is_deterministic() -> None:
    arguments = (["A", "B", "A"], ["A", "A", "A"], ["c1", "c1", "c2"], ["A", "B"])

    first = bootstrap_confidence_intervals(*arguments, iterations=20, seed=7)
    second = bootstrap_confidence_intervals(*arguments, iterations=20, seed=7)

    assert first == second
    assert first["method"] == "contract_level_percentile_bootstrap"


def test_error_records_retain_traceability() -> None:
    example = EvaluationExample("clause", "contract", "text", "A", 4, 20)

    errors = build_errors([example], ["B"], [0.9], split="test", model_name="model")

    assert errors[0]["clause_id"] == "clause"
    assert errors[0]["confidence"] == 0.9
    assert errors[0]["split"] == "test"
    summary = summarize_errors(errors, probability_confidence=True)
    assert summary["high_confidence_incorrect_count"] == 1
    assert summary["errors_by_clause_length"] == {"short_under_100": 1}
