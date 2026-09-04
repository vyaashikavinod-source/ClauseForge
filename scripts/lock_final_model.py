"""Freeze a selected candidate's identity before one-time test evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from clauseforge.artifacts.release import lock_final_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-artifact-id")
    parser.add_argument(
        "--allow-incomplete-training-selection",
        metavar="REASON",
        help="explicitly authorize an intentionally selected incomplete trajectory",
    )
    args = parser.parse_args(argv)
    lock_final_model(
        args.artifact_manifest,
        args.validation_report,
        args.output,
        datetime.now(UTC).isoformat(),
        incomplete_training_selection_reason=args.allow_incomplete_training_selection,
        selected_artifact_id=args.selected_artifact_id,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
