import numpy as np

from clauseforge.evaluation.calibration import calibration_metrics
from clauseforge.evaluation.confusion import confusion_analysis
from clauseforge.evaluation.metrics import classification_metrics, top_k_recall


def test_macro_and_weighted_metrics_differ_on_imbalanced_data() -> None:
    metrics = classification_metrics(
        ["A", "A", "A", "B"], ["A", "A", "A", "A"], ["A", "B"]
    )

    assert metrics["macro_f1"] != metrics["weighted_f1"]
    assert metrics["accuracy"] == 0.75


def test_top_k_recall_uses_ranked_scores() -> None:
    scores = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])

    assert top_k_recall(["C", "A"], scores, ["A", "B", "C"], 1) == 0.5
    assert top_k_recall(["C", "A"], scores, ["A", "B", "C"], 3) == 1.0


def test_confusion_pairs_are_sorted() -> None:
    result = confusion_analysis(["A", "A", "B", "B"], ["B", "B", "A", "B"], ["A", "B"])
    assert result["top_confusions"][0]["count"] == 2  # type: ignore[index]


def test_calibration_ece_brier_and_bins() -> None:
    result = calibration_metrics(
        ["A", "B"], np.array([[0.8, 0.2], [0.3, 0.7]]), ["A", "B"], bins=2
    )
    assert 0 <= result["ece"] <= 1  # type: ignore[operator]
    assert result["multiclass_brier_score"] > 0  # type: ignore[operator]
    assert len(result["bins"]) == 2  # type: ignore[arg-type]
