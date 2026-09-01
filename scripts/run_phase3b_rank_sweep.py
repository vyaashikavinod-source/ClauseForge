"""Plan the Qwen rank matrix or select its validation winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.training.phase3b import select_rank_winner

RANKS = (8, 16, 32, 64)


def commands(data: Path) -> list[str]:
    return [
        "python scripts/train_classifier.py "
        f"--config training/configs/phase3b/qwen25_7b_qlora_r{rank}.yaml "
        f"--data {data}"
        for rank in RANKS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        help="JSON list containing rank, split=validation, and measured macro_f1",
    )
    args = parser.parse_args(argv)
    output: dict[str, object] = {
        "ranks": list(RANKS),
        "commands": commands(args.data),
        "selection_metric": "validation macro_f1",
        "tie_break": "smaller rank",
        "test_evaluated": False,
        "executed": False,
    }
    if args.results:
        values = json.loads(args.results.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not all(
            isinstance(row, dict) for row in values
        ):
            raise ValueError("results must be a JSON list of objects")
        output["winner"] = select_rank_winner(values)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
