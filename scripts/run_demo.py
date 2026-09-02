"""Run the public-safe synthetic ClauseForge mock API demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from clauseforge.config import Settings
from clauseforge.serving.app import create_app

DEMO_LABEL = "DEVELOPMENT / SYNTHETIC DEMO — NOT MODEL PERFORMANCE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("demo/synthetic_clauses.json")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list):
        raise ValueError("demo fixtures must be a JSON list")
    selected = fixtures if args.limit is None else fixtures[: args.limit]
    results: list[dict[str, object]] = []
    with TestClient(create_app(Settings(environment="development"))) as client:
        for raw in selected:
            fixture = cast(dict[str, object], raw)
            text = str(fixture["text"])
            response = client.post("/v1/classify", json={"text": text})
            response.raise_for_status()
            results.append(
                {
                    "fixture_id": fixture["id"],
                    "input": text,
                    "expected_demo_category": fixture["expected_demo_category"],
                    "classification": response.json(),
                }
            )
    print(json.dumps({"label": DEMO_LABEL, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
