"""Deterministic metadata views over the canonical CUAD taxonomy."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from importlib.resources import files

_QUOTED_NAME = re.compile(r'related to "([^"]+)"')
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class TaxonomyCategory:
    canonical: str
    category_name: str
    category_id: str


def _category_id(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return _NON_ALNUM.sub("_", ascii_name.casefold()).strip("_")


def load_taxonomy_metadata() -> tuple[TaxonomyCategory, ...]:
    """Load all canonical strings and derive stable display metadata."""
    raw = json.loads(
        files("clauseforge").joinpath("cuad_taxonomy.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("packaged taxonomy must be a list of strings")
    categories: list[TaxonomyCategory] = []
    for canonical in raw:
        match = _QUOTED_NAME.search(canonical)
        if match is None:
            raise ValueError("canonical taxonomy string lacks a quoted category name")
        name = match.group(1)
        categories.append(TaxonomyCategory(canonical, name, _category_id(name)))
    ids = [item.category_id for item in categories]
    if len(categories) != 41 or len(set(ids)) != len(ids):
        raise ValueError("taxonomy must contain 41 unique stable category IDs")
    return tuple(categories)


def category_by_id(category_id: str) -> TaxonomyCategory:
    matches = [
        item for item in load_taxonomy_metadata() if item.category_id == category_id
    ]
    if len(matches) != 1:
        raise KeyError(category_id)
    return matches[0]


def category_by_canonical(canonical: str) -> TaxonomyCategory:
    matches = [item for item in load_taxonomy_metadata() if item.canonical == canonical]
    if len(matches) != 1:
        raise KeyError(canonical)
    return matches[0]
