"""Print a validated AWQ or GGUF plan; never converts implicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.quantization.config import load_config
from clauseforge.quantization.plans import awq_plan, gguf_plan, serialize_plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = (
        awq_plan(config)
        if config.method == "awq"
        else serialize_plans(gguf_plan(config))
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
