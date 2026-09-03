"""Validation-only ranking for replaceable trained model candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest, ValidationSummary
from clauseforge.artifacts.validation import load_manifest, validate_manifest


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    recommended_artifact_id: str
    other_artifact_id: str
    reason: str
    candidate_a: dict[str, object]
    candidate_b: dict[str, object]
    split: str = "validation"
    test_evaluated: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_candidates(first_path: Path, second_path: Path) -> CandidateComparison:
    first = _load_candidate(first_path)
    second = _load_candidate(second_path)
    first_key = _rank_key(first.validation_summary, first.checkpoint_step)
    second_key = _rank_key(second.validation_summary, second.checkpoint_step)
    winner, other = (first, second) if first_key >= second_key else (second, first)
    reason = _decision_reason(first, second)
    return CandidateComparison(
        winner.artifact_id,
        other.artifact_id,
        reason,
        _evidence(first),
        _evidence(second),
    )


def _load_candidate(path: Path) -> ArtifactManifest:
    report = validate_manifest(path, require_files=False)
    if not report.valid:
        raise ValueError("invalid candidate manifest: " + "; ".join(report.errors))
    manifest = load_manifest(path)
    summary = manifest.validation_summary
    if summary is None or summary.split != "validation":
        raise ValueError("candidate comparison requires validation metrics")
    if manifest.test_evaluated:
        raise ValueError("candidate comparison must not use held-out test evidence")
    return manifest


def _rank_key(summary: ValidationSummary | None, step: int) -> tuple[float, ...]:
    assert summary is not None
    loss = summary.validation_loss
    return (
        summary.macro_f1,
        -summary.invalid_output_rate,
        summary.exact_id_rate,
        -(loss if loss is not None else float("inf")),
        -float(step),
    )


def _evidence(manifest: ArtifactManifest) -> dict[str, object]:
    summary = manifest.validation_summary
    assert summary is not None
    return {
        "artifact_id": manifest.artifact_id,
        "checkpoint_step": manifest.checkpoint_step,
        "macro_f1": summary.macro_f1,
        "accuracy": summary.accuracy,
        "weighted_f1": summary.weighted_f1,
        "invalid_output_rate": summary.invalid_output_rate,
        "exact_id_rate": summary.exact_id_rate,
        "validation_loss": summary.validation_loss,
    }


def _decision_reason(first: ArtifactManifest, second: ArtifactManifest) -> str:
    assert (
        first.validation_summary is not None and second.validation_summary is not None
    )
    first_loss = first.validation_summary.validation_loss
    second_loss = second.validation_summary.validation_loss
    labels = (
        (
            "macro_f1",
            first.validation_summary.macro_f1,
            second.validation_summary.macro_f1,
        ),
        (
            "invalid_output_rate",
            -first.validation_summary.invalid_output_rate,
            -second.validation_summary.invalid_output_rate,
        ),
        (
            "exact_id_rate",
            first.validation_summary.exact_id_rate,
            second.validation_summary.exact_id_rate,
        ),
        (
            "validation_loss",
            -(first_loss if first_loss is not None else float("inf")),
            -(second_loss if second_loss is not None else float("inf")),
        ),
        ("earlier_checkpoint", -first.checkpoint_step, -second.checkpoint_step),
    )
    return next((name for name, left, right in labels if left != right), "equivalent")
