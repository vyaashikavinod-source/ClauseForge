"""Externally injected deterministic build metadata."""

from __future__ import annotations

import os

from clauseforge import __version__
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION


def build_metadata() -> dict[str, str | None]:
    return {
        "application_version": __version__,
        "build_commit": os.getenv("CLAUSEFORGE_BUILD_COMMIT") or None,
        "build_timestamp": os.getenv("CLAUSEFORGE_BUILD_TIMESTAMP") or None,
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
    }
