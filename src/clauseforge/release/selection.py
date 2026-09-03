"""Validation-only deterministic final-checkpoint selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class CandidateScore:
    checkpoint: str
    global_step: int
    validation_macro_f1: float
    invalid_output_rate: float
    exact_id_rate: float
    validation_loss: float


def select_final_candidate(experiment_dir: Path) -> CandidateScore:
    candidates: list[CandidateScore] = []
    for path in sorted(experiment_dir.glob("checkpoint-*/validation_metrics.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("split") not in {None, "validation"}:
            raise ValueError(f"non-validation metrics are forbidden: {path.name}")
        candidates.append(
            CandidateScore(
                str(path.parent),
                int(cast(int | str, raw["global_step"])),
                float(cast(float | str, raw["validation_macro_f1"])),
                float(cast(float | str, raw["invalid_output_rate"])),
                float(cast(float | str, raw["exact_id_rate"])),
                float(cast(float | str, raw["validation_loss"])),
            )
        )
    if not candidates:
        raise ValueError("no validation checkpoint metrics found")
    return min(
        candidates,
        key=lambda item: (
            -item.validation_macro_f1,
            item.invalid_output_rate,
            -item.exact_id_rate,
            item.validation_loss,
            item.global_step,
        ),
    )


def write_selection(path: Path, selection: CandidateScore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "clauseforge-validation-selection-v1",
                "selection_basis": "validation_only",
                "tie_breakers": [
                    "lower_invalid_output_rate",
                    "higher_exact_id_rate",
                    "lower_validation_loss",
                    "earlier_checkpoint",
                ],
                **asdict(selection),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
