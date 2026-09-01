"""Keyword baseline derived only from authoritative category names."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from clauseforge.baselines.base import FloatMatrix, StringArray

_QUOTED_CATEGORY = re.compile(r'related to "([^"]+)"', re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"a", "all", "and", "for", "of", "on", "or", "the", "to"})


def category_keywords(category: str) -> tuple[str, ...]:
    match = _QUOTED_CATEGORY.search(category)
    label = match.group(1) if match else category
    return tuple(
        token for token in _TOKEN.findall(label.lower()) if token not in _STOPWORDS
    )


class KeywordRuleClassifier:
    """Score labels by literal category-name keyword occurrences, then priors."""

    name = "rules"

    def __init__(self) -> None:
        self.classes_: StringArray = np.array([], dtype=str)
        self._keywords: list[tuple[str, ...]] = []
        self._priors: FloatMatrix = np.empty((1, 0), dtype=float)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        del texts
        if not labels:
            raise ValueError("rule classifier requires training labels")
        counts = Counter(labels)
        self.classes_ = np.array(sorted(counts), dtype=str)
        self._keywords = [category_keywords(label) for label in self.classes_]
        self._priors = np.array(
            [[counts[label] / len(labels) for label in self.classes_]], dtype=float
        )

    def predict_scores(self, texts: list[str]) -> FloatMatrix:
        if not len(self.classes_):
            raise RuntimeError("classifier is not fitted")
        scores = np.repeat(self._priors * 1e-3, len(texts), axis=0)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for column, keywords in enumerate(self._keywords):
                scores[row, column] += sum(
                    1.0
                    for keyword in keywords
                    if re.search(rf"\b{re.escape(keyword)}\b", lowered)
                )
        return scores

    def predict(self, texts: list[str]) -> StringArray:
        return self.classes_[np.argmax(self.predict_scores(texts), axis=1)]

    def predict_proba(self, texts: list[str]) -> FloatMatrix | None:
        del texts
        return None

    def configuration(self) -> dict[str, object]:
        return {
            "rules": "literal_tokens_from_quoted_authoritative_category_name",
            "tie_break": "training_class_prior_then_lexical_class_order",
        }
