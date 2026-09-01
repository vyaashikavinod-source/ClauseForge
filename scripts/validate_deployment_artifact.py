"""Validate a prepared deployment manifest and its local file checksums."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.quantization.validation import validate_deployment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = validate_deployment(args.manifest)
    print(f"deployment artifact valid: {manifest.deployment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
