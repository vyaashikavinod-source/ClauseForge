"""Validate adapter metadata and print an explicit merge plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.quantization.merge import AdapterMetadata, ModelIdentity, merge_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args(argv)
    raw = json.loads(
        (args.adapter / "adapter_metadata.json").read_text(encoding="utf-8")
    )
    adapter = AdapterMetadata(**raw)
    base = ModelIdentity(args.base_model, args.base_revision, adapter.architecture)
    print(
        json.dumps(
            merge_plan(base, adapter, str(args.output), args.build_commit),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
