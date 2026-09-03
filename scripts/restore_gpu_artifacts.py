"""Validate and safely restore a local ClauseForge GPU backup archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.archive import restore_gpu_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-experiment-id", required=True)
    args = parser.parse_args(argv)
    result = restore_gpu_artifacts(
        args.archive, args.output, expected_experiment_id=args.expected_experiment_id
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
