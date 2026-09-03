"""Run unlabeled EDGAR OOD inference through an explicitly configured real backend."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from clauseforge.config import Settings
from clauseforge.evaluation.ood import summarize_ood_predictions
from clauseforge.serving.constants import CUAD_TAXONOMY
from clauseforge.serving.dependencies import build_provider


async def run(input_path: Path, output: Path) -> dict[str, object]:
    settings = Settings.from_env()
    if settings.model_provider == "mock":
        raise ValueError("final EDGAR OOD refuses the mock backend")
    if settings.model_artifact_manifest is None:
        raise ValueError("final EDGAR OOD requires MODEL_ARTIFACT_MANIFEST")
    provider = build_provider(settings, CUAD_TAXONOMY)
    rows: list[dict[str, object]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)
        if (
            not isinstance(source, dict)
            or "text" not in source
            or "segment_id" not in source
        ):
            raise ValueError("invalid EDGAR segment record")
        try:
            result = await provider.classify(str(source["text"]))
            rows.append(
                {
                    "segment_id": str(source["segment_id"]),
                    "prediction": result.category,
                    "confidence": max(result.scores.values())
                    if result.scores
                    else None,
                    "abstained": result.category is None,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "segment_id": str(source["segment_id"]),
                    "prediction": None,
                    "confidence": None,
                    "abstained": True,
                    "error": type(exc).__name__,
                }
            )
    await provider.close()
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize_ood_predictions(rows, frozenset(CUAD_TAXONOMY))
    summary.update(
        {
            "label": "FINAL EDGAR OOD — UNLABELED; NO ACCURACY CLAIM",
            "artifact_manifest": str(settings.model_artifact_manifest),
            "predictions": predictions.name,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(asyncio.run(run(args.input, args.output)), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
