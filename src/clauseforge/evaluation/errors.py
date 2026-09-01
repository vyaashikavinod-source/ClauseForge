"""Machine-readable prediction errors."""

from __future__ import annotations

from collections import Counter
from typing import cast

from clauseforge.evaluation.dataset import EvaluationExample


def build_errors(
    examples: list[EvaluationExample],
    predictions: list[str],
    confidences: list[float | None],
    *,
    split: str,
    model_name: str,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = [
        {
            "clause_id": example.clause_id,
            "contract_id": example.contract_id,
            "true_category": example.label,
            "predicted_category": prediction,
            "confidence": confidence,
            "clause_text": example.text,
            "clause_length": example.clause_length,
            "contract_length": example.contract_length,
            "split": split,
            "model_name": model_name,
        }
        for example, prediction, confidence in zip(
            examples, predictions, confidences, strict=True
        )
        if prediction != example.label
    ]
    return sorted(errors, key=lambda item: str(item["clause_id"]))


def summarize_errors(
    errors: list[dict[str, object]], *, probability_confidence: bool
) -> dict[str, object]:
    def clause_bucket(length: int) -> str:
        if length < 100:
            return "short_under_100"
        if length < 500:
            return "medium_100_to_499"
        return "long_500_plus"

    def contract_bucket(length: int) -> str:
        if length < 25_000:
            return "short_under_25000"
        if length < 100_000:
            return "medium_25000_to_99999"
        return "long_100000_plus"

    by_class = Counter(str(error["true_category"]) for error in errors)
    by_clause_length = Counter(
        clause_bucket(cast(int, error["clause_length"])) for error in errors
    )
    by_contract_length = Counter(
        contract_bucket(cast(int, error["contract_length"])) for error in errors
    )
    high_confidence = (
        sum(
            error["confidence"] is not None and cast(float, error["confidence"]) >= 0.8
            for error in errors
        )
        if probability_confidence
        else None
    )
    return {
        "error_count": len(errors),
        "errors_by_true_class": dict(sorted(by_class.items())),
        "errors_by_clause_length": dict(sorted(by_clause_length.items())),
        "errors_by_contract_length": dict(sorted(by_contract_length.items())),
        "high_confidence_incorrect_count": high_confidence,
        "high_confidence_threshold": 0.8 if probability_confidence else None,
    }
