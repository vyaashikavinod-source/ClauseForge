"""Run or validate an existing locked real-model safety/OOD report."""

import argparse
import asyncio
import json
from pathlib import Path

from clauseforge.release.final_checks import run_check, validate_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["safety", "ood"], required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.output / "summary.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else asyncio.run(
            run_check(
                args.kind, args.lock, args.artifact_manifest, args.output, args.input
            )
        )
    )
    validate_check(report, args.kind, args.lock, args.artifact_manifest)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
