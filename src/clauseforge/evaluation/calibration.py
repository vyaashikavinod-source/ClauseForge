"""Multiclass probability calibration diagnostics."""

from __future__ import annotations

import numpy as np

from clauseforge.baselines.base import FloatMatrix


def calibration_metrics(
    y_true: list[str], probabilities: FloatMatrix, classes: list[str], *, bins: int = 10
) -> dict[str, object]:
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("probability matrix shape mismatch")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.array([class_index[label] for label in y_true])
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, object]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        average_confidence = float(confidence[mask].mean()) if count else 0.0
        accuracy = float(correctness[mask].mean()) if count else 0.0
        ece += count / len(y_true) * abs(accuracy - average_confidence)
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "average_confidence": average_confidence,
            }
        )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return {
        "ece": float(ece),
        "multiclass_brier_score": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "bins": reliability,
    }
