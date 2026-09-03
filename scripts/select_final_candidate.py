"""Select a checkpoint using validation metrics only and deterministic ties."""

from __future__ import annotations

import argparse
from pathlib import Path

from clauseforge.release.selection import select_final_candidate, write_selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = select_final_candidate(args.experiment_dir)
    write_selection(args.output, selected)
    print(selected.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
