"""Conservative generative adapter for the Phase 2 classifier contract."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from clauseforge.baselines.base import FloatMatrix, StringArray
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION, render_prompt


class UnknownGeneratedLabelError(ValueError):
    """Raised instead of silently coercing a hallucinated label."""


def normalize_generated_label(value: str, taxonomy: tuple[str, ...]) -> str:
    candidate = value.strip()
    if candidate not in taxonomy:
        raise UnknownGeneratedLabelError(
            f"generated value is not a CUAD label: {candidate!r}"
        )
    return candidate


class TransformerClassifier:
    """Adapter around an exact-label text generator.

    Ranked scores are intentionally unavailable until a calibrated label-scoring
    strategy is implemented and evaluated.
    """

    name = "transformer-causal-classifier"

    def __init__(
        self, taxonomy: tuple[str, ...], generator: Callable[[str], str]
    ) -> None:
        self._taxonomy = taxonomy
        self._generator = generator
        self.classes_: StringArray = np.asarray(taxonomy, dtype=np.str_)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        unknown = set(labels) - set(self._taxonomy)
        if unknown:
            raise ValueError(
                f"training view contains unknown labels: {sorted(unknown)}"
            )

    def predict(self, texts: list[str]) -> StringArray:
        return np.asarray(
            [
                normalize_generated_label(
                    self._generator(render_prompt(text)), self._taxonomy
                )
                for text in texts
            ],
            dtype=np.str_,
        )

    def predict_scores(self, texts: list[str]) -> FloatMatrix:
        raise NotImplementedError("ranked class scores are not genuinely available")

    def predict_proba(self, texts: list[str]) -> FloatMatrix | None:
        return None

    def configuration(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "label_normalization": "exact_strip_only",
            "ranked_scores": False,
        }
