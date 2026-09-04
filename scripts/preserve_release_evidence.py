"""Snapshot explicit release evidence, never datasets, weights, or credentials."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from clauseforge.artifacts.release import validate_final_lock
from clauseforge.artifacts.validation import sha256_file


def preserve(
    manifest: Path, lock: Path, evidence: Path, output: Path
) -> dict[str, object]:
    identity = validate_final_lock(lock, manifest)
    files = {"artifact_manifest.json": manifest, "final_lock.json": lock}
    receipt = lock.with_name(lock.name + ".held-out-attempt.json")
    if receipt.exists():
        files["final_lock.json.held-out-attempt.json"] = receipt
    for name in (
        "test-command.json",
        "safety-command.json",
        "ood-command.json",
        "environment.json",
        "held-out-test.json",
        "held-out-predictions.jsonl",
        "final-evaluation-bundle.json",
        "safety/summary.json",
        "safety/adversarial_results.jsonl",
        "safety/paraphrase_results.jsonl",
        "safety/taxonomy_results.json",
        "safety/run_config.json",
        "ood/summary.json",
        "ood/predictions.jsonl",
    ):
        path = evidence / name
        if path.is_file():
            files[f"evidence/{name}"] = path
    for path in files.values():
        if path.is_symlink():
            raise ValueError("evidence symlinks are not supported")
    receipt_data: dict[str, object] = {
        "schema_version": "clauseforge-release-evidence-backup-v1",
        "artifact_id": identity.artifact_id,
        "selection_checksum": identity.selection_checksum,
        "files": {name: sha256_file(path) for name, path in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        payload = json.dumps(receipt_data, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("checksums.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
    return receipt_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-manifest", "lock", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            preserve(args.artifact_manifest, args.lock, args.evidence, args.output),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
