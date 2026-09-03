"""Validate an external adapter for registration without copying model data."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.artifacts.workflows import serialize_import, validate_import


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(serialize_import(validate_import(args.adapter, args.manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
