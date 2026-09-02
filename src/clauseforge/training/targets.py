"""Versioned exact model-facing target representations."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Literal

from clauseforge.taxonomy import TaxonomyCategory, load_taxonomy_metadata

TargetRepresentation = Literal["canonical_question", "category_id"]
CANONICAL_TARGET_VERSION = "cuad-canonical-question-v1"
CATEGORY_ID_TARGET_VERSION = "cuad-category-id-v1"


class UnknownTargetError(ValueError):
    """Raised when generated output is not an exact member of its target set."""


@lru_cache(maxsize=1)
def stable_id_map() -> tuple[TaxonomyCategory, ...]:
    return load_taxonomy_metadata()


def stable_id_map_checksum() -> str:
    payload = [
        {
            "canonical": item.canonical,
            "category_id": item.category_id,
            "category_name": item.category_name,
        }
        for item in stable_id_map()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def target_for_canonical(canonical: str, representation: TargetRepresentation) -> str:
    matches = [item for item in stable_id_map() if item.canonical == canonical]
    if len(matches) != 1:
        raise UnknownTargetError("unknown canonical taxonomy target")
    return (
        canonical if representation == "canonical_question" else matches[0].category_id
    )


def resolve_generated_target(
    raw: str, representation: TargetRepresentation
) -> TaxonomyCategory:
    """Strip outer whitespace only, then perform exact target lookup."""
    candidate = raw.strip()
    for item in stable_id_map():
        expected = (
            item.canonical
            if representation == "canonical_question"
            else item.category_id
        )
        if candidate == expected:
            return item
    raise UnknownTargetError("generated output is not an exact valid target")


def validate_target_version(
    representation: TargetRepresentation, version: str, prompt_version: str
) -> None:
    expected = {
        "canonical_question": (CANONICAL_TARGET_VERSION, "cuad-classification-v1"),
        "category_id": (CATEGORY_ID_TARGET_VERSION, "cuad-classification-id-v2"),
    }[representation]
    if (version, prompt_version) != expected:
        raise ValueError("target representation and prompt versions are incompatible")
