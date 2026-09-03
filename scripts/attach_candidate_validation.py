"""Attach validation-only evidence to a generated candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.candidate import attach_validation_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = attach_validation_evidence(
        args.manifest, args.validation_evidence, args.output
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
