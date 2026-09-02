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
    parser.add_argument(
        "--pilot-validation-examples",
        type=int,
        help="override persisted validation subset size for a new diagnostic sample",
    )
    parser.add_argument("--diagnostics", action="store_true", required=True)
    parser.add_argument(
        "--debug-validation-examples",
        type=int,
        help="evaluate 4-8 validation examples and print safe output diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.debug_validation_examples is not None and not (
        4 <= args.debug_validation_examples <= 8
    ):
        raise ValueError("debug validation example count must be between 4 and 8")
    result = evaluate_phase3b_checkpoint(
        load_config(args.config),
        args.data,
        args.checkpoint,
        args.output,
        pilot=args.pilot,
        pilot_validation_examples=(
            args.debug_validation_examples
            if args.debug_validation_examples is not None
            else args.pilot_validation_examples
        ),
    )
    if args.debug_validation_examples is not None:
        rows = []
        predictions = args.output / "validation_predictions.jsonl"
        for line in predictions.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            rows.append(
                {
                    "clause_id": record["clause_id"],
                    "target_id": record["canonical_target_id"],
                    "generated_output": record["raw_generated_text"],
                    "diagnostic_status": record["status_reason"],
                }
            )
        result["safe_validation_debug"] = rows
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
