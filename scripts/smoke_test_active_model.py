"""Run synthetic-only real-model checks; never a performance evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from clauseforge.config import Settings
from clauseforge.serving.constants import CUAD_TAXONOMY
from clauseforge.serving.dependencies import build_provider
from clauseforge.training.targets import resolve_generated_target

SYNTHETIC = (
    "This agreement is governed by the laws of Delaware.",
    "Either party may terminate for convenience with sixty days written notice.",
    "The company may audit relevant records once each calendar year.",
)


async def run() -> dict[str, object]:
    settings = Settings.from_env()
    if settings.model_provider not in {"real", "transformer"}:
        raise ValueError("smoke test requires CLAUSEFORGE_MODEL_BACKEND=real")
    provider = build_provider(settings, CUAD_TAXONOMY)
    ready, detail = provider.is_ready()
    if not ready:
        raise ValueError(detail or "real model is unavailable")
    rows = []
    try:
        for text in SYNTHETIC:
            started = time.perf_counter()
            result = await provider.classify(text)
            target = resolve_generated_target(result.category or "", "category_id")
            rows.append(
                {
                    "input": text,
                    "predicted_category_id": target.category_id,
                    "canonical_category": target.canonical,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    finally:
        await provider.close()
    return {
        "label": "REAL MODEL SMOKE TEST — NOT FINAL MODEL PERFORMANCE",
        "artifact_id": settings.artifact_id,
        "checkpoint_step": settings.checkpoint_step,
        "test_evaluated": False,
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(asyncio.run(run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
