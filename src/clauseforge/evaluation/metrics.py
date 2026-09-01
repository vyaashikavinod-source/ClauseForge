"""Multiclass metrics and ranked recall."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from clauseforge.baselines.base import FloatMatrix


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, object]:
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            label: {
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index, label in enumerate(labels)
        },
    }


def top_k_recall(
    y_true: list[str], scores: FloatMatrix, classes: list[str], k: int
) -> float:
    if scores.shape != (len(y_true), len(classes)):
        raise ValueError("score matrix shape does not match examples and classes")
    effective_k = min(k, len(classes))
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :effective_k]
    hits = sum(
        true_label in {classes[index] for index in row}
        for true_label, row in zip(y_true, ranked, strict=True)
    )
    return hits / len(y_true) if y_true else 0.0
