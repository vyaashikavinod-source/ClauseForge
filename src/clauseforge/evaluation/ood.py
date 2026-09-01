"""Metrics appropriate for unlabeled out-of-distribution predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean, median


def summarize_ood_predictions(
    predictions: Sequence[Mapping[str, object]], taxonomy: frozenset[str]
) -> dict[str, object]:
    """Summarize outputs without fabricating accuracy or ground-truth labels."""
    total = len(predictions)
    valid = [row for row in predictions if str(row.get("prediction", "")) in taxonomy]
    confidences: list[float] = []
    for row in valid:
        confidence = row.get("confidence")
        if isinstance(confidence, int | float):
            confidences.append(float(confidence))
    distribution = Counter(str(row["prediction"]) for row in valid)
    failures = sum(bool(row.get("error")) for row in predictions)
    abstentions = sum(bool(row.get("abstained")) for row in predictions)
    return {
        "record_count": total,
        "taxonomy_valid_output_rate": len(valid) / total if total else 0.0,
        "invalid_output_rate": (total - len(valid)) / total if total else 0.0,
        "prediction_distribution": dict(sorted(distribution.items())),
        "confidence_distribution": (
            {
                "available": True,
                "count": len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "mean": mean(confidences),
                "median": median(confidences),
            }
            if confidences
            else {"available": False, "reason": "provider did not expose real scores"}
        ),
        "abstention_rate": abstentions / total if total else 0.0,
        "processing_failures": failures,
        "ground_truth_metrics_available": False,
    }
