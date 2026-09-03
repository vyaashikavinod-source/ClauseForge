"""Explicitly authorize deployment after every final release gate passes."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.artifacts.release import authorize_deployment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--authorize-deployment", action="store_true", required=True)
    args = parser.parse_args(argv)
    manifest = authorize_deployment(args.artifact_manifest)
    print(f"deployment authorized: {manifest.artifact_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
