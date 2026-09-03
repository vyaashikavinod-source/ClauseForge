"""Freeze a fully trained candidate's identity before one-time test evaluation."""

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
    args = parser.parse_args(argv)
    lock_final_model(
        args.artifact_manifest,
        args.validation_report,
        args.output,
        datetime.now(UTC).isoformat(),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
