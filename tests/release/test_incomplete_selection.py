from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.lock_final_model import main as lock_main
from scripts.run_final_validation import main as final_validation_main

from clauseforge.artifacts.comparison import compare_candidates
from clauseforge.artifacts.registry import ArtifactRegistry, promotion_gates
from clauseforge.artifacts.release import (
    authorize_test_once,
    lock_final_model,
    mark_manifest_test_evaluated,
    record_test_evaluated,
    validate_final_lock,
)
from clauseforge.artifacts.validation import load_manifest, sha256_path, write_manifest
from clauseforge.artifacts.workflows import final_validation_plan
from clauseforge.evaluation.locked_test import (
    evaluate_locked_test,
    verify_authorization,
)
from clauseforge.release.status import build_release_status
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.training.targets import stable_id_map

REASON = "Operator intentionally selected checkpoint 800 using validation only"


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = replace(
        load_manifest(Path("configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json")),
        artifact_id="selected-800",
        checkpoint_step=800,
        full_training_completed=False,
        validation_selection_completed=True,
    )
    adapter = tmp_path / "checkpoint-800"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"offline fixture only")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": manifest.lora_rank,
                "lora_alpha": manifest.lora_alpha,
                "target_modules": manifest.lora_targets,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "global_step": 800,
                "experiment_id": manifest.adapter_experiment_id,
                "prompt_template_version": manifest.prompt_version,
                "target_representation": manifest.target_representation,
                "target_representation_version": manifest.target_representation_version,
                "stable_id_map_checksum": manifest.stable_id_map_checksum,
                "lora_rank": manifest.lora_rank,
                "lora_alpha": manifest.lora_alpha,
                "target_modules": manifest.lora_targets,
            }
        ),
        encoding="utf-8",
    )
    manifest = replace(
        manifest,
        adapter_path=str(adapter),
        adapter_checksum=sha256_path(adapter),
        required_files={},
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    summary = manifest.validation_summary
    assert summary is not None
    report = tmp_path / "validation.json"
    report.write_text(
        json.dumps(
            {
                "split": "validation",
                "test_evaluated": False,
                "checkpoint_step": 800,
                "experiment_id": manifest.adapter_experiment_id,
                "prompt_version": manifest.prompt_version,
                "target_representation": manifest.target_representation,
                "target_representation_version": manifest.target_representation_version,
                "stable_id_map_checksum": manifest.stable_id_map_checksum,
                "accuracy": summary.accuracy,
                "macro_f1": summary.macro_f1,
                "weighted_f1": summary.weighted_f1,
                "invalid_output_rate": summary.invalid_output_rate,
                "validation_loss": summary.validation_loss,
                "total_examples": summary.validation_examples,
                "exact_id_output_rate": summary.exact_id_rate,
            }
        ),
        encoding="utf-8",
    )
    return path, report, tmp_path / "lock.json"


def test_incomplete_requires_explicit_authorization(
    candidate: tuple[Path, Path, Path],
) -> None:
    path, report, output = candidate
    with pytest.raises(ValueError, match="explicit release authorization"):
        lock_final_model(path, report, output, "now")
    write_manifest(path, replace(load_manifest(path), validation_summary=None))
    with pytest.raises(ValueError, match="validation"):
        lock_final_model(path, report, output, "now")


def test_explicit_lock_preserves_history_and_one_time_test(
    candidate: tuple[Path, Path, Path],
) -> None:
    path, report, output = candidate
    original = path.read_bytes()
    lock = lock_final_model(
        path,
        report,
        output,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    assert path.read_bytes() == original
    assert not lock.full_training_completed
    assert lock.checkpoint_step == 800
    assert lock.incomplete_training_selection_reason == REASON
    assert lock.validation_report_checksum and lock.selection_checksum
    assert (
        build_release_status(
            path, lock_path=output
        ).incomplete_training_selection_reason
        == REASON
    )
    authorize_test_once(output)
    assert (
        final_validation_plan(path, authorize_test=True, lock_path=output)[
            "test_authorized"
        ]
        is True
    )
    with pytest.raises(ValueError, match="already"):
        authorize_test_once(output)
    record_test_evaluated(output)
    mark_manifest_test_evaluated(path)
    assert (
        validate_final_lock(output, path).incomplete_training_selection_reason == REASON
    )
    assert not load_manifest(path).full_training_completed
    with pytest.raises(ValueError, match="already"):
        record_test_evaluated(output)
    with pytest.raises(ValueError, match="already"):
        authorize_test_once(output)
    with pytest.raises(ValueError, match="already"):
        final_validation_plan(path, authorize_test=True, lock_path=output)
    with pytest.raises(ValueError, match="already exists"):
        lock_final_model(path, report, output, "later")
    with pytest.raises(ValueError, match="held-out test"):
        compare_candidates(path, path)


@pytest.mark.parametrize(
    "change",
    [
        {"test_evaluated": True},
        {"validation_selection_completed": False},
        {"release_status": "pilot"},
        {"adapter_checksum": "0" * 64},
    ],
)
def test_rejects_unqualified_selection(
    candidate: tuple[Path, Path, Path], change: dict[str, object]
) -> None:
    path, report, output = candidate
    raw = load_manifest(path).to_dict()
    raw.update(change)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        lock_final_model(
            path,
            report,
            output,
            "now",
            incomplete_training_selection_reason=REASON,
            selected_artifact_id="selected-800",
        )


def test_wrong_selected_id_and_evidence_rejected(
    candidate: tuple[Path, Path, Path],
) -> None:
    path, report, output = candidate
    with pytest.raises(ValueError, match="fixed"):
        lock_final_model(
            path,
            report,
            output,
            "now",
            incomplete_training_selection_reason=REASON,
            selected_artifact_id="other",
        )
    raw = json.loads(report.read_text(encoding="utf-8"))
    raw["split"] = "test"
    report.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        lock_final_model(
            path,
            report,
            output,
            "now",
            incomplete_training_selection_reason=REASON,
            selected_artifact_id="selected-800",
        )


def test_promotion_exception_does_not_bypass_other_gates(
    candidate: tuple[Path, Path, Path],
) -> None:
    path, report, output = candidate
    lock = lock_final_model(
        path,
        report,
        output,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    manifest = load_manifest(path)
    gates = promotion_gates(manifest, "final_candidate", lock=lock)
    assert "full training completed" not in gates.blocking_gates
    assert "final safety evaluation completed" in gates.blocking_gates
    assert "final EDGAR OOD evaluation completed" in gates.blocking_gates
    assert "held-out test evaluated once" in gates.blocking_gates
    completed = replace(
        manifest,
        test_evaluated=True,
        final_safety_evaluated=True,
        final_ood_evaluated=True,
    )
    assert not promotion_gates(completed, "final_candidate").allowed
    assert promotion_gates(completed, "final_candidate", lock=lock).allowed
    write_manifest(path, completed)
    registry = ArtifactRegistry(path.parent / "registry")
    registry.register(path)
    promoted = registry.promote(
        manifest.artifact_id, "final_candidate", lock_path=output
    )
    assert not promoted.full_training_completed
    assert not promotion_gates(promoted, "released", lock=lock).allowed
    write_manifest(path, replace(completed, checkpoint_step=801))
    with pytest.raises(ValueError, match="lineage"):
        validate_final_lock(output, path)


def test_fully_trained_unchanged(candidate: tuple[Path, Path, Path]) -> None:
    path, report, output = candidate
    write_manifest(path, replace(load_manifest(path), full_training_completed=True))
    lock = lock_final_model(path, report, output, "now")
    assert lock.full_training_completed
    assert lock.incomplete_training_selection_reason is None


def test_cli_explicit_authorization(candidate: tuple[Path, Path, Path]) -> None:
    path, report, output = candidate
    args = [
        "--artifact-manifest",
        str(path),
        "--validation-report",
        str(report),
        "--output",
        str(output),
        "--selected-artifact-id",
        "selected-800",
    ]
    with pytest.raises(ValueError, match="explicit release authorization"):
        lock_main(args)
    assert lock_main([*args, "--allow-incomplete-training-selection", REASON]) == 0
    assert (
        validate_final_lock(output, path).incomplete_training_selection_reason == REASON
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_revision", "changed"),
        ("lora_rank", 16),
        ("full_training_completed", True),
        ("adapter_checksum", "f" * 64),
    ],
)
def test_locked_selection_cannot_change(
    candidate: tuple[Path, Path, Path],
    field: str,
    value: object,
) -> None:
    path, report, output = candidate
    lock_final_model(
        path,
        report,
        output,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    raw = load_manifest(path).to_dict()
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage"):
        validate_final_lock(output, path)


def test_final_bundle_preserves_exception(
    candidate: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, report, output = candidate
    lock_final_model(
        path,
        report,
        output,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    authorize_test_once(output)
    command = path.parent / "command.json"
    command.write_text('["offline-fixture"]', encoding="utf-8")
    result = path.parent / "fixture-result.json"
    _test_evidence(path, output, result, monkeypatch)
    calls: list[list[str]] = []

    def offline_run(args: list[str], *, check: bool, shell: bool) -> None:
        assert check and not shell
        calls.append(args)

    monkeypatch.setattr("scripts.run_final_validation.subprocess.run", offline_run)
    bundle = path.parent / "bundle.json"
    args = [
        "--artifact-manifest",
        str(path),
        "--lock",
        str(output),
        "--authorize-held-out-test",
        "--execute",
        "--environment-report",
        str(result),
        "--output-bundle",
        str(bundle),
    ]
    for phase in ("test", "safety", "ood"):
        args.extend(
            [f"--{phase}-command-json", str(command), f"--{phase}-report", str(result)]
        )
    assert final_validation_main(args) == 0
    assert len(calls) == 3
    assert (
        json.loads(bundle.read_text(encoding="utf-8"))[
            "incomplete_training_selection_reason"
        ]
        == REASON
    )
    assert not load_manifest(path).full_training_completed
    with pytest.raises(ValueError, match="already"):
        final_validation_main(args)
    assert len(calls) == 3


class OfflineTestProvider:
    name = "offline-fixture"
    model_id = "offline-fixture"
    provider_type = "transformer"
    is_mock = False  # Test double only; not exposed by the production CLI.

    def is_ready(self) -> tuple[bool, str | None]:
        return True, None

    async def classify(self, text: str) -> ProviderResult:
        assert text.startswith("synthetic held-out")
        raw = stable_id_map()[0].category_id if text.endswith("1") else ""
        return ProviderResult(raw, raw, None)

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("failure", ["not_ready", "mock"])
def test_held_out_refuses_unavailable_or_mock_provider(
    candidate: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path, evidence, lock = candidate
    lock_final_model(
        path,
        evidence,
        lock,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    authorize_test_once(lock)
    provider = OfflineTestProvider()
    if failure == "mock":
        provider.is_mock = True
    else:
        monkeypatch.setattr(provider, "is_ready", lambda: (False, None))
    monkeypatch.setattr(
        "clauseforge.evaluation.locked_test._provider", lambda *args: provider
    )

    def forbidden(*args: object) -> None:
        pytest.fail("test data accessed without a real ready provider")

    monkeypatch.setattr("clauseforge.evaluation.locked_test._load_test_only", forbidden)
    with pytest.raises(ValueError, match="no fallback"):
        asyncio.run(
            evaluate_locked_test(
                lock,
                path,
                path.parent / "never-open",
                path.parent / "out.json",
                path.parent / "pred.jsonl",
            )
        )


def _test_evidence(
    path: Path, lock: Path, output: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    data = path.parent / "synthetic-data"
    data.mkdir()
    (data / "splits.json").write_text(
        json.dumps(
            {"train": ["train"], "validation": ["validation"], "test": ["test"]}
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "clause_id": name,
            "contract_id": split,
            "text": text,
            "category": stable_id_map()[0].canonical,
        }
        for name, split, text in [
            ("train", "train", "never infer training"),
            ("val", "validation", "never infer validation"),
            ("test1", "test", "synthetic held-out 1"),
            ("test2", "test", "synthetic held-out 2"),
        ]
    ]
    (data / "clauses.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        "clauseforge.evaluation.locked_test._provider",
        lambda *args: OfflineTestProvider(),
    )
    return asyncio.run(
        evaluate_locked_test(
            lock, path, data, output, path.parent / "test-predictions.jsonl"
        )
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unauthorized",
        "evaluated",
        "identity",
        "adapter",
        "checkpoint",
        "manifest_checksum",
    ],
)
def test_held_out_rejects_before_data_access(
    candidate: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path, evidence, lock = candidate
    lock_final_model(
        path,
        evidence,
        lock,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    if failure != "unauthorized":
        authorize_test_once(lock)
    if failure == "missing":
        lock = path.parent / "missing-lock.json"
    elif failure == "evaluated":
        record_test_evaluated(lock)
    elif failure == "identity":
        write_manifest(path, replace(load_manifest(path), artifact_id="wrong"))
    elif failure == "checkpoint":
        write_manifest(path, replace(load_manifest(path), checkpoint_step=700))
    elif failure == "adapter":
        (path.parent / "checkpoint-800" / "adapter_model.safetensors").write_bytes(
            b"corrupted fixture"
        )
    elif failure == "manifest_checksum":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def forbidden(*args: object) -> None:
        pytest.fail("unauthorized path reached dataset/model loader")

    monkeypatch.setattr("clauseforge.evaluation.locked_test._load_test_only", forbidden)
    monkeypatch.setattr("clauseforge.evaluation.locked_test._provider", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        asyncio.run(
            evaluate_locked_test(
                lock,
                path,
                path.parent / "never-open",
                path.parent / "test.json",
                path.parent / "predictions.jsonl",
            )
        )


def test_held_out_report_and_read_only_state(
    candidate: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, evidence, lock = candidate
    lock_final_model(
        path,
        evidence,
        lock,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    authorize_test_once(lock)
    original = (path.read_bytes(), lock.read_bytes())
    output = path.parent / "held-out.json"
    result = _test_evidence(path, lock, output, monkeypatch)
    assert (path.read_bytes(), lock.read_bytes()) == original
    assert result["split"] == "test"
    assert result["label"] == "HELD-OUT FINAL TEST"
    assert result["total_examples"] == 2
    assert result["exact_id_count"] == 1
    assert result["invalid_output_count"] == 1
    assert result["empty_output_count"] == 1
    assert result["accuracy"] == 0.5
    assert result["test_loss"] is None
    assert result["test_dataset_checksum"]
    assert result["test_evaluated"] is True  # Synthetic evidence, not runtime state.
    assert result["incomplete_training_selection_reason"] == REASON
    rows = [
        json.loads(line)
        for line in (path.parent / "test-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["clause_id"] for row in rows} == {"test1", "test2"}
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(
            evaluate_locked_test(
                lock,
                path,
                path.parent / "synthetic-data",
                path.parent / "new-report.json",
                path.parent / "new-predictions.jsonl",
            )
        )
    assert verify_authorization(lock, path)[0].test_authorized


@pytest.mark.parametrize("failure", ["command", "json", "schema"])
def test_orchestrator_failure_does_not_consume_test(
    candidate: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path, evidence, lock = candidate
    lock_final_model(
        path,
        evidence,
        lock,
        "now",
        incomplete_training_selection_reason=REASON,
        selected_artifact_id="selected-800",
    )
    authorize_test_once(lock)
    command = path.parent / "command.json"
    command.write_text('["offline-fixture"]', encoding="utf-8")
    report = path.parent / "report.json"
    report.write_text("bad json" if failure == "json" else "{}", encoding="utf-8")

    def fake_run(args: list[str], *, check: bool, shell: bool) -> None:
        if failure == "command":
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr("scripts.run_final_validation.subprocess.run", fake_run)
    args = [
        "--artifact-manifest",
        str(path),
        "--lock",
        str(lock),
        "--authorize-held-out-test",
        "--execute",
        "--environment-report",
        str(report),
        "--output-bundle",
        str(path.parent / "bundle.json"),
    ]
    for phase in ("test", "safety", "ood"):
        args.extend(
            [f"--{phase}-command-json", str(command), f"--{phase}-report", str(report)]
        )
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        final_validation_main(args)
    assert not load_manifest(path).test_evaluated
    assert verify_authorization(lock, path)[0].test_authorized
