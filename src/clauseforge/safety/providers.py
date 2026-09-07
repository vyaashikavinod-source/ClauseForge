"""CPU-safe provider adapters for safety evaluation."""

from __future__ import annotations

from clauseforge.baselines.rules import KeywordRuleClassifier
from clauseforge.serving.providers.base import ProviderResult


class ClassicalRuleProvider:
    name = "classical-keyword-rules"
    model_id = "phase2-keyword-rules-equal-priors"
    provider_type = "mock"
    is_mock = True
    target_representation = "canonical_question"
    target_representation_version = "cuad-canonical-question-v1"

    def __init__(self, taxonomy: tuple[str, ...]) -> None:
        self._classifier = KeywordRuleClassifier()
        self._classifier.fit([""] * len(taxonomy), list(taxonomy))

    def is_ready(self) -> tuple[bool, str | None]:
        return True, None

    async def classify(self, text: str) -> ProviderResult:
        category = str(self._classifier.predict([text])[0])
        scores = self._classifier.predict_scores([text])[0]
        ranked = {
            str(label): float(score)
            for label, score in zip(self._classifier.classes_, scores, strict=True)
        }
        return ProviderResult(category, category, ranked)

    async def close(self) -> None:
        return None
