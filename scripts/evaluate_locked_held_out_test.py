"""Evaluate an authorized locked transformer on the final test split exactly once."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from clauseforge.evaluation.locked_test import evaluate_locked_test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    asyncio.run(
        evaluate_locked_test(
            args.lock,
            args.artifact_manifest,
            args.data,
            args.output,
            args.predictions,
            max_sequence_length=args.max_sequence_length,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
