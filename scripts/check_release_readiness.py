"""Perform an offline ClauseForge release-readiness assessment."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.config import Settings
from clauseforge.release.readiness import assess_readiness, serialize_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--quantization-manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args(argv)
    report = assess_readiness(
        Settings.from_env(),
        deployment_manifest=args.deployment_manifest,
        quantization_manifest=args.quantization_manifest,
        lock_path=args.lock,
    )
    print(serialize_report(report))
    return 1 if args.strict and report.overall_status != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
