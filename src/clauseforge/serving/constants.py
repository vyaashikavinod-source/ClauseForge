"""Versioned product and taxonomy metadata."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

DISCLAIMER = (
    "This system assists with contract clause analysis and does not provide "
    "legal advice."
)
TAXONOMY_VERSION = "cuad-v1-41"


def load_taxonomy() -> tuple[str, ...]:
    value = json.loads(
        files("clauseforge").joinpath("cuad_taxonomy.json").read_text(encoding="utf-8")
    )
    if not isinstance(value, list) or len(value) != 41:
        raise RuntimeError("packaged CUAD taxonomy must contain exactly 41 labels")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise RuntimeError("packaged CUAD taxonomy contains an invalid label")
    return tuple(cast(list[str], value))


CUAD_TAXONOMY = load_taxonomy()
