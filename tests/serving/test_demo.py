from __future__ import annotations

import json
from pathlib import Path

from scripts.run_demo import DEMO_LABEL, main


def test_demo_fixtures_are_synthetic_and_complete() -> None:
    fixtures = json.loads(
        Path("demo/synthetic_clauses.json").read_text(encoding="utf-8")
    )
    assert len(fixtures) == 8
    assert len({item["id"] for item in fixtures}) == 8
    assert all(
        set(item) == {"id", "text", "expected_demo_category", "notes"}
        for item in fixtures
    )
    assert all("synthetic" in item["notes"].casefold() for item in fixtures[:2])


def test_demo_cli_uses_mock_schema(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--limit", "1"]) == 0
    output = json.loads(capsys.readouterr().out)
    result = output["results"][0]["classification"]
    assert output["label"] == DEMO_LABEL
    assert result["provider"] == "mock-development"
    assert result["request_id"] and result["taxonomy_version"]
    assert result["processing"]["is_mock"] is True
    assert result["disclaimer"]
