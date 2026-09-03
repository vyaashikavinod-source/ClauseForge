"""Validate a ClauseForge model artifact manifest entirely offline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from clauseforge.artifacts.validation import validate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    report = validate_manifest(args.manifest, require_files=not args.metadata_only)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
