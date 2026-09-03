"""Report the complete local ClauseForge release state without network or GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.release.status import (
    build_release_status,
    human_status,
    serialize_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json"),
    )
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_release_status(args.manifest, lock_path=args.lock)
    print(serialize_status(report) if args.json else human_status(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
