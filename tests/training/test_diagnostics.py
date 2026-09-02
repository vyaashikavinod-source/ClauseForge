from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from clauseforge.models.transformer_classifier import (
    UnknownGeneratedLabelError,
    normalize_generated_label,
)
from clauseforge.taxonomy import category_by_id, load_taxonomy_metadata
from clauseforge.training.config import load_config
from clauseforge.training.diagnostics import (
    ValidationPrediction,
    aggregate_diagnostics,
    build_prediction,
    write_predictions,
)


def _prediction(raw: str, *, tokens: int = 2, limit: int = 160) -> ValidationPrediction:
    renewal = category_by_id("renewal_term")
    return build_prediction(
        clause_id="validation-clause-1",
        canonical_target=renewal.canonical,
        raw_generated_text=raw,
        generated_token_count=tokens,
        target_token_count=48,
        generation_limit=limit,
    )


def test_short_name_is_diagnostic_only_and_exact_validation_stays_strict() -> None:
    prediction = _prediction("Renewal Term")
    assert prediction.status_reason == "short_name_only"
    assert prediction.validation_status == "invalid" and not prediction.exact_match
    with pytest.raises(UnknownGeneratedLabelError):
        normalize_generated_label(
            "Renewal Term", tuple(x.canonical for x in load_taxonomy_metadata())
        )


def test_exact_commentary_and_truncation_diagnostics() -> None:
    canonical = category_by_id("renewal_term").canonical
    exact = _prediction(canonical)
    whitespace_changed = _prediction(
        canonical.replace("Highlight the", "Highlight  the")
    )
    commentary = _prediction(f"The category is {canonical}")
    truncated = _prediction(canonical[:80], tokens=160, limit=160)
    assert exact.exact_match and exact.validation_status == "exact"
    assert not whitespace_changed.exact_match
    assert commentary.status_reason == "commentary_wrapped"
    assert commentary.validation_status == "malformed"
    assert truncated.status_reason == "truncated_output"
    assert truncated.validation_status == "invalid"


def test_prediction_artifact_serialization_and_aggregates(tmp_path: Path) -> None:
    canonical = category_by_id("renewal_term").canonical
    records = [
        _prediction(canonical),
        _prediction("Renewal Term"),
        _prediction(f"prefix {canonical} suffix"),
        _prediction(""),
        _prediction("unrelated answer"),
    ]
    output = tmp_path / "validation_predictions.jsonl"
    write_predictions(output, records)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert set(rows[0]) == {
        "clause_id",
        "canonical_target",
        "canonical_target_id",
        "canonical_target_name",
        "raw_generated_text",
        "normalized_generated_text",
        "validation_status",
        "status_reason",
        "exact_match",
        "generated_token_count",
        "target_token_count",
        "target_representation",
    }
    assert all(row["clause_id"].startswith("validation-") for row in rows)
    counts = aggregate_diagnostics(records)
    assert counts["exact_canonical_matches"] == 1
    assert counts["short_name_only_outputs"] == 1
    assert counts["canonical_substring_outputs"] == 1
    assert counts["empty_outputs"] == 1
    assert counts["completely_unrelated_outputs"] == 1


def test_generation_limit_is_explicit_experiment_metadata() -> None:
    config = load_config(Path("training/configs/phase3b/qwen25_7b_qlora_r8.yaml"))
    assert config.data.validation_max_new_tokens == 160
    data = cast(dict[str, object], config.to_dict()["data"])
    assert data["validation_max_new_tokens"] == 160
