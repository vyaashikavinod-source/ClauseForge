"""Versioned safety-fixture records and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["low", "medium", "high"]
ExpectedBehavior = Literal["taxonomy_valid", "reject_request", "ignore_instruction"]


@dataclass(frozen=True, slots=True)
class SafetyCase:
    case_id: str
    category: str
    source: str
    text: str
    expected_behavior: ExpectedBehavior
    expected_category: str | None
    attack_type: str
    notes: str
    severity: Severity
    provenance: str


@dataclass(frozen=True, slots=True)
class ParaphrasePair:
    pair_id: str
    source: str
    first: str
    second: str
    notes: str
    provenance: str
