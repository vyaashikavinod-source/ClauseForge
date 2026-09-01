"""Word TF-IDF linear classifiers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from clauseforge.baselines.base import FloatMatrix, StringArray


class TfidfLinearClassifier:
    """TF-IDF with logistic regression or a linear support-vector classifier."""

    def __init__(
        self,
        kind: Literal["logreg", "svm"],
        *,
        c: float = 1.0,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ) -> None:
        self.kind = kind
        self.name = f"tfidf-{kind}"
        self.c = c
        self.random_state = random_state
        self.class_weight = class_weight
        estimator = (
            LogisticRegression(
                C=c,
                class_weight=class_weight,
                max_iter=2000,
                random_state=random_state,
                solver="lbfgs",
            )
            if kind == "logreg"
            else LinearSVC(C=c, class_weight=class_weight, random_state=random_state)
        )
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        sublinear_tf=True,
                        max_features=100_000,
                        strip_accents="unicode",
                    ),
                ),
                ("classifier", estimator),
            ]
        )
        self.classes_: StringArray = np.array([], dtype=str)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.pipeline.fit(texts, labels)
        classifier = self.pipeline.named_steps["classifier"]
        self.classes_ = np.asarray(classifier.classes_, dtype=str)

    def predict(self, texts: list[str]) -> StringArray:
        return np.asarray(self.pipeline.predict(texts), dtype=str)

    def predict_scores(self, texts: list[str]) -> FloatMatrix:
        if self.kind == "logreg":
            return np.asarray(self.pipeline.predict_proba(texts), dtype=float)
        scores = np.asarray(self.pipeline.decision_function(texts), dtype=float)
        if scores.ndim == 1:
            return np.column_stack((-scores, scores))
        return scores

    def predict_proba(self, texts: list[str]) -> FloatMatrix | None:
        if self.kind != "logreg":
            return None
        return np.asarray(self.pipeline.predict_proba(texts), dtype=float)

    def configuration(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "C": self.c,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "tfidf": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.995,
                "max_features": 100_000,
                "sublinear_tf": True,
                "strip_accents": "unicode",
            },
        }
