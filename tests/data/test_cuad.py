from __future__ import annotations

import json
from pathlib import Path

import pytest

from clauseforge.data.cuad import CuadFormatError, load_cuad


def test_loads_squad_style_cuad_and_preserves_answers(fixture_path: Path) -> None:
    dataset = load_cuad(fixture_path)

    assert dataset.version == "1.0"
    assert len(dataset.documents) == 2
    assert dataset.documents[0].paragraphs[0].questions[1].answers[0].text == "New York"


def test_missing_file_has_descriptive_error(tmp_path: Path) -> None:
    with pytest.raises(CuadFormatError, match="does not exist"):
        load_cuad(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda root: root.pop("data"), "root.data"),
        (lambda root: root["data"][0].pop("title"), "title"),
        (
            lambda root: root["data"][0]["paragraphs"][0]["qas"][0]["answers"][
                0
            ].update({"answer_start": "33"}),
            "answer_start",
        ),
    ],
)
def test_malformed_source_fails_safely(
    fixture_path: Path, tmp_path: Path, mutation: object, expected: str
) -> None:
    root = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(root)
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(CuadFormatError, match=expected):
        load_cuad(malformed)


def test_empty_annotation_is_rejected(fixture_path: Path, tmp_path: Path) -> None:
    root = json.loads(fixture_path.read_text(encoding="utf-8"))
    root["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"] = ""
    malformed = tmp_path / "empty.json"
    malformed.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(CuadFormatError, match="non-empty string"):
        load_cuad(malformed)
