"""Restore a checksum-indexed release snapshot without overwriting current state."""

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.archive import restore_release_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(restore_release_evidence(args.archive, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
