"""Idempotent offline preparation; never infers missing historical evidence."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from clauseforge.artifacts.candidate import (
    attach_validation_evidence,
    create_candidate_manifest,
)
from clauseforge.artifacts.registry import ArtifactRegistry
from clauseforge.artifacts.release import (
    authorize_test_once,
    lock_final_model,
    selection_checksum,
    validate_final_lock,
)
from clauseforge.artifacts.validation import (
    load_manifest,
    sha256_file,
    validate_manifest,
)
from clauseforge.release.final_checks import validate_bundle

ARTIFACT = "clauseforge-qwen25-r8-step800-recovery-v1"
TRAINING_COMMIT = "a96fbab8c4f96a69bc40ad64ae1eee1354bab08e"


def write_once(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"existing prepared file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def environment_report() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu_results": None,
        "label": "PREPARATION ENVIRONMENT — NOT GPU BENCHMARK EVIDENCE",
    }


def prepare(
    checkpoint: Path,
    manifest: Path,
    registry_root: Path,
    lock_path: Path,
    validation_report: Path,
    data: Path,
    edgar: Path,
    output: Path,
    *,
    initialize_reason: str | None = None,
    authorize: bool = False,
) -> dict[str, object]:
    # No datasets are opened here. Missing artifacts stop before any mutation.
    for path in (checkpoint, data):
        if not path.is_dir():
            raise ValueError(f"missing directory: {path}")
    for path in (
        validation_report,
        edgar,
        checkpoint / "checkpoint_metadata.json",
        checkpoint / "resume_state.json",
        checkpoint.parent / "experiment_config.json",
    ):
        if not path.is_file():
            raise ValueError(f"missing historical artifact: {path}")
    state = json.loads((checkpoint / "resume_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != 800:
        raise ValueError("handoff requires selected checkpoint 800")
    receipt = lock_path.with_name(lock_path.name + ".held-out-attempt.json")
    if not lock_path.exists() and (receipt.exists() or not initialize_reason):
        raise ValueError(
            "restore the original lock/attempt state; initialization requires "
            "explicit fresh-selection authorization"
        )
    if not manifest.exists():
        if lock_path.exists():
            raise ValueError(
                "locked manifest missing: restore exact bytes, do not recreate"
            )
        create_candidate_manifest(checkpoint, ARTIFACT, manifest, TRAINING_COMMIT)
        attach_validation_evidence(manifest, validation_report, manifest)
    selected = load_manifest(manifest)
    if (
        selected.artifact_id != ARTIFACT
        or selected.checkpoint_step != 800
        or not validate_manifest(manifest).valid
    ):
        raise ValueError("selected checkpoint artifact mismatch")
    if (
        manifest.parent / (selected.adapter_path or "")
    ).resolve() != checkpoint.resolve():
        raise ValueError("selected adapter path mismatch")
    if lock_path.exists():
        lock = validate_final_lock(lock_path, manifest)
        if lock.validation_report_checksum != sha256_file(validation_report):
            raise ValueError("historical validation evidence checksum mismatch")
    else:
        from datetime import UTC, datetime

        lock = lock_final_model(
            manifest,
            validation_report,
            lock_path,
            datetime.now(UTC).isoformat(),
            incomplete_training_selection_reason=initialize_reason,
            selected_artifact_id=ARTIFACT,
        )
    if lock.test_evaluated != selected.test_evaluated or (
        lock.test_evaluated and lock.test_authorized
    ):
        raise ValueError("inconsistent one-time state; stop for recovery review")
    if receipt.exists() and not lock.test_evaluated:
        raise ValueError("prior held-out attempt is unconsumed; do not retry")
    registry = ArtifactRegistry(registry_root)
    registered = registry.manifests / f"{ARTIFACT}.json"
    if registered.exists():
        if selection_checksum(load_manifest(registered)) != lock.selection_checksum:
            raise ValueError(
                "registered manifest differs; restore exact selected state"
            )
    else:
        registry.register(manifest)
    if registry.active_path.exists():
        # Evaluation updates gate flags in-place: accept only the same selected ID
        # and validate current manifest, but do not silently repair a stale pointer.
        pointer = json.loads(registry.active_path.read_text(encoding="utf-8"))
        if pointer.get("current_artifact_id") != ARTIFACT:
            raise ValueError("another candidate is active")
        if lock.test_evaluated and registered.resolve() == manifest.resolve():
            if pointer.get(
                "current_manifest"
            ) != f"manifests/{ARTIFACT}.json" or pointer.get(
                "current_manifest_checksum"
            ) not in {lock.manifest_checksum, sha256_file(manifest)}:
                raise ValueError("active pointer is not the locked selection")
            # Refresh only the checksum of the same locked ID after gate updates.
            pointer["current_manifest_checksum"] = sha256_file(manifest)
            registry.active_path.write_text(
                json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        registry.read_active()
    else:
        registry.activate(ARTIFACT)
    if not lock.test_evaluated and not lock.test_authorized:
        if not authorize:
            raise ValueError("explicit --authorize-held-out-test required")
        lock = authorize_test_once(lock_path)
    common = ["--lock", str(lock_path), "--artifact-manifest", str(manifest)]
    commands = {
        "test-command.json": [
            sys.executable,
            "scripts/evaluate_locked_held_out_test.py",
            *common,
            "--data",
            str(data),
            "--output",
            str(output / "held-out-test.json"),
            "--predictions",
            str(output / "held-out-predictions.jsonl"),
        ],
        "safety-command.json": [
            sys.executable,
            "scripts/run_locked_release_check.py",
            *common,
            "--kind",
            "safety",
            "--output",
            str(output / "safety"),
        ],
        "ood-command.json": [
            sys.executable,
            "scripts/run_locked_release_check.py",
            *common,
            "--kind",
            "ood",
            "--input",
            str(edgar),
            "--output",
            str(output / "ood"),
        ],
    }
    for name, argv in commands.items():
        write_once(output / name, argv)
    if not (output / "environment.json").exists():
        write_once(output / "environment.json", environment_report())
    argv = [
        sys.executable,
        "scripts/run_final_validation.py",
        *common,
        "--execute",
        "--resume-after-test" if lock.test_evaluated else "--authorize-held-out-test",
    ]
    for kind, report in (
        ("test", "held-out-test.json"),
        ("safety", "safety/summary.json"),
        ("ood", "ood/summary.json"),
    ):
        argv.extend(
            [
                f"--{kind}-command-json",
                str(output / f"{kind}-command.json"),
                f"--{kind}-report",
                str(output / report),
            ]
        )
    argv.extend(
        [
            "--environment-report",
            str(output / "environment.json"),
            "--output-bundle",
            str(output / "final-evaluation-bundle.json"),
        ]
    )
    bundle = output / "final-evaluation-bundle.json"
    if bundle.exists():
        validate_bundle(
            json.loads(bundle.read_text(encoding="utf-8")), lock_path, manifest
        )
    return {
        "artifact_id": ARTIFACT,
        "test_evaluated": lock.test_evaluated,
        "executed": False,
        "next_argv": argv,
        "bundle_complete": bundle.exists(),
        "gpu_required": True,
    }
