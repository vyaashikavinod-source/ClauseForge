"""Compare two model candidates using validation evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.comparison import compare_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            compare_candidates(args.candidate_a, args.candidate_b).to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
