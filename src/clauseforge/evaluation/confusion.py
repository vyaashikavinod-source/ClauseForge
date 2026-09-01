"""Confusion-matrix summaries."""

from __future__ import annotations

from typing import cast

from sklearn.metrics import confusion_matrix


def confusion_analysis(
    y_true: list[str], y_pred: list[str], labels: list[str], *, limit: int = 20
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, object]] = [
        {
            "true": labels[row],
            "predicted": labels[column],
            "count": int(matrix[row, column]),
        }
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and matrix[row, column] > 0
    ]
    pairs.sort(
        key=lambda item: (
            -cast(int, item["count"]),
            str(item["true"]),
            str(item["predicted"]),
        )
    )
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "top_confusions": pairs[:limit],
    }
