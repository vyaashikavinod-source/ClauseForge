"""Package allowlisted GPU outputs locally; never upload or include datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.archive import package_gpu_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            package_gpu_artifacts(args.experiment, args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
