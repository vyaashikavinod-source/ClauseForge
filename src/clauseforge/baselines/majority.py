"""Empirical class-prior baseline."""

from __future__ import annotations

from collections import Counter

import numpy as np

from clauseforge.baselines.base import FloatMatrix, StringArray


class MajorityClassifier:
    name = "majority"

    def __init__(self) -> None:
        self.classes_: StringArray = np.array([], dtype=str)
        self._probabilities: FloatMatrix = np.empty((1, 0), dtype=float)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        del texts
        if not labels:
            raise ValueError("majority classifier requires training labels")
        counts = Counter(labels)
        self.classes_ = np.array(sorted(counts), dtype=str)
        total = len(labels)
        self._probabilities = np.array(
            [[counts[label] / total for label in self.classes_]], dtype=float
        )

    def predict(self, texts: list[str]) -> StringArray:
        self._check_fitted()
        winner = self.classes_[int(np.argmax(self._probabilities[0]))]
        return np.asarray([winner] * len(texts), dtype=str)

    def predict_scores(self, texts: list[str]) -> FloatMatrix:
        self._check_fitted()
        return np.repeat(self._probabilities, len(texts), axis=0)

    def predict_proba(self, texts: list[str]) -> FloatMatrix:
        return self.predict_scores(texts)

    def configuration(self) -> dict[str, object]:
        return {"strategy": "most_frequent", "ranking": "training_class_priors"}

    def _check_fitted(self) -> None:
        if not len(self.classes_):
            raise RuntimeError("classifier is not fitted")
