"""Record explicit one-time test authorization for one locked candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.artifacts.release import authorize_test_once, validate_final_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--authorize-held-out-test", action="store_true", required=True)
    args = parser.parse_args(argv)
    validate_final_lock(args.lock, args.artifact_manifest)
    lock = authorize_test_once(args.lock)
    print(f"authorized once (not yet evaluated): {lock.artifact_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
