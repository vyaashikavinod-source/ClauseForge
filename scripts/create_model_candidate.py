"""Create a candidate manifest from any compatible restored checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.candidate import create_candidate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    args = parser.parse_args(argv)
    manifest = create_candidate_manifest(
        args.checkpoint, args.artifact_id, args.output, args.training_commit
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
