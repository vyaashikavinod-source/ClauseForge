"""Strict local loaders for synthetic, versioned safety fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from clauseforge.safety.models import ParaphrasePair, SafetyCase


class SafetyFixtureError(ValueError):
    """Raised when a committed safety fixture is malformed."""


def _records(path: Path, version: str) -> list[dict[str, Any]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyFixtureError(f"unable to read safety fixture: {path.name}") from exc
    if not isinstance(root, dict) or root.get("version") != version:
        raise SafetyFixtureError(f"fixture must use version {version}")
    records = root.get("cases")
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise SafetyFixtureError("fixture cases must be a list of objects")
    return cast(list[dict[str, Any]], records)


def load_safety_cases(path: Path) -> tuple[SafetyCase, ...]:
    try:
        cases = tuple(SafetyCase(**record) for record in _records(path, "safety-v1"))
    except TypeError as exc:
        raise SafetyFixtureError(
            "safety case has missing or unexpected fields"
        ) from exc
    if len({case.case_id for case in cases}) != len(cases):
        raise SafetyFixtureError("safety case IDs must be unique")
    if any(case.source != "synthetic-public-safe" for case in cases):
        raise SafetyFixtureError("only synthetic public-safe cases are permitted")
    return cases


def load_paraphrase_pairs(path: Path) -> tuple[ParaphrasePair, ...]:
    try:
        pairs = tuple(
            ParaphrasePair(**record) for record in _records(path, "paraphrase-v1")
        )
    except TypeError as exc:
        raise SafetyFixtureError("paraphrase pair has invalid fields") from exc
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise SafetyFixtureError("paraphrase pair IDs must be unique")
    return pairs
