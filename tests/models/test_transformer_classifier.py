from __future__ import annotations

from pathlib import Path

import pytest

from clauseforge.baselines.base import ClassifierProtocol
from clauseforge.evaluation.dataset import EvaluationDataset, EvaluationExample
from clauseforge.evaluation.runner import evaluate_model
from clauseforge.models.transformer_classifier import (
    TransformerClassifier,
    UnknownGeneratedLabelError,
    normalize_generated_label,
)


def test_generated_label_validation_and_protocol() -> None:
    model = TransformerClassifier(("A", "B"), lambda _prompt: " A\n")
    assert isinstance(model, ClassifierProtocol)
    assert model.predict(["clause"]).tolist() == ["A"]
    assert model.predict_proba(["clause"]) is None
    with pytest.raises(NotImplementedError):
        model.predict_scores(["clause"])


def test_unknown_label_is_rejected_not_coerced() -> None:
    with pytest.raises(UnknownGeneratedLabelError):
        normalize_generated_label("Probably A", ("A", "B"))


def test_phase2_runner_supports_honest_absence_of_scores(tmp_path: Path) -> None:
    model = TransformerClassifier(("A", "B"), lambda _prompt: "A")
    example = EvaluationExample("c1", "d1", "clause", "A", 6, 6)
    dataset = EvaluationDataset(
        {"train": [example], "validation": [example], "test": [example]},
        ["A", "B"],
        tmp_path,
    )
    summary = evaluate_model(
        model, dataset, "validation", tmp_path / "evaluation", bootstrap_iterations=10
    )
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["accuracy"] == 1.0
    assert metrics["top_k_recall"] == {
        "available": False,
        "reason": "model does not expose ranked scores",
    }
