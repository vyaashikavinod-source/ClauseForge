"""Strict prediction records with non-accepting output diagnostics."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from clauseforge.taxonomy import TaxonomyCategory, load_taxonomy_metadata
from clauseforge.training.targets import TargetRepresentation

_SPACE = re.compile(r"\s+")
_COMMENTARY = re.compile(
    r"^(?:the (?:answer|category|label) is|category:|label:|answer:)", re.I
)


@dataclass(frozen=True, slots=True)
class ValidationPrediction:
    clause_id: str
    canonical_target: str
    canonical_target_id: str
    canonical_target_name: str
    raw_generated_text: str
    normalized_generated_text: str
    validation_status: str
    status_reason: str
    exact_match: bool
    generated_token_count: int
    target_token_count: int
    target_representation: str = "canonical_question"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_diagnostic_text(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def diagnose_output(
    raw: str,
    target: TaxonomyCategory,
    *,
    generated_token_count: int,
    generation_limit: int,
) -> tuple[str, str]:
    """Classify output shape without changing strict taxonomy acceptance."""
    value = normalize_diagnostic_text(raw)
    if raw.strip() == target.canonical:
        return "exact", "exact_canonical_match"
    if not value:
        return "empty", "empty_output"
    if value.casefold() == target.category_name.casefold():
        return "invalid", "short_name_only"
    if target.canonical.startswith(value) and len(value) < len(target.canonical):
        reason = (
            "truncated_output"
            if generated_token_count >= generation_limit
            else "canonical_prefix"
        )
        return "invalid", reason
    if target.canonical in value:
        return (
            "malformed" if "\n" in raw or _COMMENTARY.search(value) else "invalid",
            "commentary_wrapped"
            if _COMMENTARY.search(value)
            else "canonical_substring",
        )
    if "\n" in raw or raw.lstrip().startswith(("'", '"', "```")):
        return "malformed", "malformed_output"
    return "invalid", "completely_unrelated"


def diagnose_id_output(raw: str, target: TaxonomyCategory) -> tuple[str, str, bool]:
    """Diagnose ID output without promoting recognizable invalid forms."""
    candidate = raw.strip()
    categories = load_taxonomy_metadata()
    exact_ids = {item.category_id: item for item in categories}
    if candidate in exact_ids:
        return "exact", "exact_id_match", candidate == target.category_id
    if not candidate:
        return "empty", "empty", False
    if candidate in {item.category_name for item in categories}:
        return "invalid", "short_name_output", False
    if candidate in {item.canonical for item in categories}:
        return "invalid", "canonical_question_output", False
    if target.category_id in candidate and candidate != target.category_id:
        return "malformed", "commentary_wrapped_id", False
    if "\n" in raw or raw.lstrip().startswith(("'", '"', "```")):
        return "malformed", "malformed", False
    if re.fullmatch(r"[a-z][a-z0-9_]*", candidate):
        return "invalid", "invalid_id", False
    return "invalid", "unrelated", False


def build_prediction(
    *,
    clause_id: str,
    canonical_target: str,
    raw_generated_text: str,
    generated_token_count: int,
    target_token_count: int,
    generation_limit: int,
    target_representation: TargetRepresentation = "canonical_question",
) -> ValidationPrediction:
    categories = load_taxonomy_metadata()
    if target_representation == "category_id":
        target = next(
            item for item in categories if item.category_id == canonical_target
        )
        status, reason, exact_match = diagnose_id_output(raw_generated_text, target)
        authoritative_target = target.canonical
    else:
        target = next(item for item in categories if item.canonical == canonical_target)
        status, reason = diagnose_output(
            raw_generated_text,
            target,
            generated_token_count=generated_token_count,
            generation_limit=generation_limit,
        )
        exact_match = status == "exact"
        authoritative_target = canonical_target
    return ValidationPrediction(
        clause_id=clause_id,
        canonical_target=authoritative_target,
        canonical_target_id=target.category_id,
        canonical_target_name=target.category_name,
        raw_generated_text=raw_generated_text,
        normalized_generated_text=normalize_diagnostic_text(raw_generated_text),
        validation_status=status,
        status_reason=reason,
        exact_match=exact_match,
        generated_token_count=generated_token_count,
        target_token_count=target_token_count,
        target_representation=target_representation,
    )


def aggregate_diagnostics(
    predictions: list[ValidationPrediction],
) -> dict[str, int]:
    reasons = Counter(item.status_reason for item in predictions)
    statuses = Counter(item.validation_status for item in predictions)
    return {
        "exact_canonical_matches": reasons["exact_canonical_match"],
        "short_name_only_outputs": reasons["short_name_only"],
        "canonical_prefix_outputs": reasons["canonical_prefix"],
        "canonical_substring_outputs": reasons["canonical_substring"],
        "commentary_wrapped_outputs": reasons["commentary_wrapped"],
        "truncated_outputs": reasons["truncated_output"],
        "empty_outputs": reasons["empty_output"],
        "malformed_outputs": statuses["malformed"],
        "completely_unrelated_outputs": reasons["completely_unrelated"],
        "exact_id_matches": reasons["exact_id_match"],
        "short_name_outputs": reasons["short_name_output"],
        "canonical_question_outputs": reasons["canonical_question_output"],
        "commentary_wrapped_ids": reasons["commentary_wrapped_id"],
        "invalid_ids": reasons["invalid_id"],
        "unrelated_outputs": reasons["unrelated"],
    }


def write_predictions(path: Path, predictions: list[ValidationPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction.to_dict(), sort_keys=True) + "\n")
