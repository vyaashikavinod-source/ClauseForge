"""List, inspect, validate, activate, rollback, or promote local artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from clauseforge.artifacts.registry import ArtifactRegistry
from clauseforge.artifacts.validation import validate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("artifact_id")
    validate = sub.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    activate = sub.add_parser("activate")
    activate.add_argument("artifact_id")
    sub.add_parser("rollback")
    promote = sub.add_parser("promote")
    promote.add_argument("artifact_id")
    promote.add_argument(
        "status", choices=["release_candidate", "final_candidate", "released"]
    )
    args = parser.parse_args(argv)
    registry = ArtifactRegistry(args.registry)
    if args.command == "list":
        result: object = [item.to_dict() for item in registry.list()]
    elif args.command == "inspect":
        result = registry.inspect(args.artifact_id).to_dict()
    elif args.command == "validate":
        result = asdict(validate_manifest(args.manifest))
    elif args.command == "activate":
        result = registry.activate(args.artifact_id).to_dict()
    elif args.command == "rollback":
        result = registry.rollback().to_dict()
    else:
        result = registry.promote(args.artifact_id, args.status).to_dict()  # type: ignore[arg-type]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
