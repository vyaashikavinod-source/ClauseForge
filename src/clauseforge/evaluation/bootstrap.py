"""Deterministic contract-level bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _metrics_from_confusion(matrix: NDArray[np.int64]) -> tuple[float, float, float]:
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=true_support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total else 0.0
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.dot(f1, true_support) / total) if total else 0.0
    return accuracy, macro_f1, weighted_f1


def bootstrap_confidence_intervals(
    y_true: list[str],
    y_pred: list[str],
    contract_ids: list[str],
    labels: list[str],
    *,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    if not (len(y_true) == len(y_pred) == len(contract_ids)):
        raise ValueError("bootstrap inputs must have equal length")
    class_index = {label: index for index, label in enumerate(labels)}
    groups: dict[str, NDArray[np.int64]] = {}
    for true_label, predicted_label, contract_id in zip(
        y_true, y_pred, contract_ids, strict=True
    ):
        matrix = groups.setdefault(
            contract_id, np.zeros((len(labels), len(labels)), dtype=np.int64)
        )
        matrix[class_index[true_label], class_index[predicted_label]] += 1
    contracts = sorted(groups)
    contract_matrices = np.stack([groups[contract] for contract in contracts])
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
    }
    for _ in range(iterations):
        sampled = rng.integers(0, len(contracts), size=len(contracts))
        accuracy, macro_f1, weighted_f1 = _metrics_from_confusion(
            contract_matrices[sampled].sum(axis=0)
        )
        values["accuracy"].append(accuracy)
        values["macro_f1"].append(macro_f1)
        values["weighted_f1"].append(weighted_f1)
    return {
        "method": "contract_level_percentile_bootstrap",
        "confidence_level": 0.95,
        "iterations": iterations,
        "seed": seed,
        "metrics": {
            name: {
                "lower": float(np.percentile(samples, 2.5)),
                "upper": float(np.percentile(samples, 97.5)),
            }
            for name, samples in values.items()
        },
    }
