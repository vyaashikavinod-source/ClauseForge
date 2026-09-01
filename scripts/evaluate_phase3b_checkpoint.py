"""Inspect Phase 3B checkpoint predictions using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.training.config import load_config
from clauseforge.training.phase3b import evaluate_phase3b_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-validation-examples", type=int, default=256)
    parser.add_argument("--diagnostics", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_phase3b_checkpoint(
        load_config(args.config),
        args.data,
        args.checkpoint,
        args.output,
        pilot=args.pilot,
        pilot_validation_examples=args.pilot_validation_examples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
