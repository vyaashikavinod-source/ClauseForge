"""Deterministic classical classification baselines."""

from clauseforge.baselines.majority import MajorityClassifier
from clauseforge.baselines.rules import KeywordRuleClassifier
from clauseforge.baselines.tfidf import TfidfLinearClassifier

__all__ = ["KeywordRuleClassifier", "MajorityClassifier", "TfidfLinearClassifier"]
