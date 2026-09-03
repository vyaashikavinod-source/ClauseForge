"""Print a non-executing final-validation plan; never opens datasets itself."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.artifacts.workflows import final_validation_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--authorize-held-out-test", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            final_validation_plan(
                args.artifact_manifest, authorize_test=args.authorize_held_out_test
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
