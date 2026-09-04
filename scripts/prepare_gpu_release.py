"""Prepare checkpoint-800 release commands without running GPU evaluation."""

import argparse
import json
from pathlib import Path

from clauseforge.release.handoff import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "manifest",
        "registry",
        "lock",
        "validation-report",
        "data",
        "edgar",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--initialize-selection", metavar="REASON")
    parser.add_argument("--authorize-held-out-test", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(
        args.checkpoint,
        args.manifest,
        args.registry,
        args.lock,
        args.validation_report,
        args.data,
        args.edgar,
        args.output,
        initialize_reason=args.initialize_selection,
        authorize=args.authorize_held_out_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
