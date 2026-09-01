import numpy as np

from clauseforge.baselines.base import ClassifierProtocol
from clauseforge.baselines.majority import MajorityClassifier
from clauseforge.baselines.rules import KeywordRuleClassifier, category_keywords
from clauseforge.baselines.tfidf import TfidfLinearClassifier


def _training() -> tuple[list[str], list[str]]:
    texts = [
        "governed by New York law",
        "laws of California govern",
        "the agreement terminates",
        "termination requires notice",
    ]
    labels = ["Governing Law", "Governing Law", "Termination", "Termination"]
    return texts, labels


def test_majority_uses_training_prior_and_probabilities() -> None:
    model = MajorityClassifier()
    model.fit([], ["Governing Law", "Termination", "Governing Law"])

    assert model.predict(["x", "y"]).tolist() == ["Governing Law", "Governing Law"]
    assert np.allclose(model.predict_proba(["x"]).sum(axis=1), 1.0)


def test_rule_baseline_uses_authoritative_category_name_tokens() -> None:
    labels = [
        'Highlight related to "Governing Law" that should be reviewed',
        'Highlight related to "Termination For Convenience" that should be reviewed',
    ]
    model = KeywordRuleClassifier()
    model.fit([], labels)

    assert category_keywords(labels[0]) == ("governing", "law")
    assert model.predict(["This provision states the governing law."])[0] == labels[0]
    assert model.predict_proba(["text"]) is None


def test_tfidf_models_fit_predict_and_score() -> None:
    texts, labels = _training()
    for kind in ("logreg", "svm"):
        model = TfidfLinearClassifier(kind, class_weight=None)
        model.pipeline.set_params(tfidf__min_df=1)
        model.fit(texts, labels)
        assert len(model.predict(texts)) == 4
        assert model.predict_scores(texts).shape == (4, 2)
        assert (model.predict_proba(texts) is not None) == (kind == "logreg")
        assert isinstance(model, ClassifierProtocol)
