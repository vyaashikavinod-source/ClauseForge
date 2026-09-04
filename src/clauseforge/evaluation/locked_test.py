"""Release-only held-out boundary. Never imported by training or selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.release import FinalModelLock, validate_final_lock
from clauseforge.artifacts.validation import (
    load_manifest,
    sha256_file,
    validate_manifest,
)
from clauseforge.evaluation.metrics import classification_metrics
from clauseforge.serving.providers.base import ClauseClassifierProvider
from clauseforge.serving.providers.local_transformer import LocalTransformerProvider
from clauseforge.training.diagnostics import diagnose_id_output
from clauseforge.training.targets import stable_id_map

REPORT_SCHEMA = "clauseforge-held-out-final-test-v1"


def validate_test_report(
    report: dict[str, object], lock_path: Path, manifest_path: Path
) -> None:
    """Check successful-command evidence before the orchestrator consumes state."""
    lock, manifest = verify_authorization(lock_path, manifest_path)
    expected = {
        "schema_version": REPORT_SCHEMA,
        "label": "HELD-OUT FINAL TEST",
        "split": "test",
        "test_evaluated": True,
        "is_mock": False,
        "artifact_id": manifest.artifact_id,
        "checkpoint_step": manifest.checkpoint_step,
        "experiment_id": manifest.adapter_experiment_id,
        "adapter_checksum": manifest.adapter_checksum,
        "manifest_checksum": lock.manifest_checksum,
        "selection_checksum": lock.selection_checksum,
        "base_model": manifest.base_model,
        "base_revision": manifest.base_revision,
        "prompt_version": manifest.prompt_version,
        "target_representation": manifest.target_representation,
        "target_representation_version": manifest.target_representation_version,
        "taxonomy_version": manifest.taxonomy_version,
        "stable_id_map_checksum": manifest.stable_id_map_checksum,
        "incomplete_training_selection_reason": (
            lock.incomplete_training_selection_reason
        ),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("held-out report identity/schema mismatch")
    for key in (
        "test_dataset_checksum",
        "split_assignments_checksum",
        "predictions_checksum",
    ):
        value = report.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("held-out report is missing evidence checksums")
    for key in ("category_coverage", "prediction_distribution", "generation"):
        if not isinstance(report.get(key), dict):
            raise ValueError("held-out report is missing diagnostics")
    total = report.get("total_examples")
    exact = report.get("exact_id_count")
    invalid = report.get("invalid_output_count")
    if (
        type(total) is not int
        or total <= 0
        or type(exact) is not int
        or type(invalid) is not int
        or exact < 0
        or invalid < 0
        or exact + invalid != total
    ):
        raise ValueError("held-out report counts are invalid")
    for key in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "exact_id_output_rate",
        "invalid_output_rate",
    ):
        value = report.get(key)
        if (
            not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("held-out report metrics are invalid")
    if not math.isclose(
        float(cast(float, report["exact_id_output_rate"])), exact / total
    ) or not math.isclose(
        float(cast(float, report["invalid_output_rate"])), invalid / total
    ):
        raise ValueError("held-out report count/rate mismatch")


def verify_authorization(
    lock_path: Path, manifest_path: Path
) -> tuple[FinalModelLock, ArtifactManifest]:
    """No dataset operations are permitted before this function succeeds."""
    lock = validate_final_lock(lock_path, manifest_path)
    manifest = load_manifest(manifest_path)
    if not lock.test_authorized or lock.test_evaluated or manifest.test_evaluated:
        raise ValueError("held-out test is unauthorized or already evaluated")
    if (
        lock.schema_version != "clauseforge-final-lock-v1"
        or not lock.selection_checksum
        or lock.manifest_checksum != sha256_file(manifest_path)
        or lock.checkpoint_step != manifest.checkpoint_step
        or not lock.adapter_checksum
        or lock.adapter_checksum != manifest.adapter_checksum
        or manifest.artifact_type != "adapter"
        or manifest.release_status not in {"release_candidate", "final_candidate"}
        or not manifest.adapter_path
    ):
        raise ValueError(
            "locked manifest/checkpoint identity mismatch or incomplete lock"
        )
    if (
        manifest.base_model != "Qwen/Qwen2.5-7B-Instruct"
        or manifest.base_revision != "a09a35458c702b33eeacc393d103063234e8bc28"
        or manifest.prompt_version != "cuad-classification-id-v2"
        or manifest.target_representation != "category_id"
        or manifest.target_representation_version != "cuad-category-id-v1"
    ):
        raise ValueError("unsupported locked production model or target path")
    validation = validate_manifest(manifest_path)
    if not validation.valid:
        raise ValueError(
            "locked artifact verification failed: " + "; ".join(validation.errors)
        )
    return lock, manifest


@dataclass(frozen=True)
class TestExample:
    clause_id: str
    contract_id: str
    text: str
    category_id: str


def _load_test_only(data: Path) -> tuple[list[TestExample], str, str]:
    """Stream the existing shared clause file; retain only sealed test contracts."""
    raw = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"train", "validation", "test"}:
        raise ValueError("invalid split assignments")
    seen: set[str] = set()
    for ids in raw.values():
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("split contract IDs must be strings")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("duplicate or overlapping split contracts")
        seen.update(ids)
    test_ids = set(raw["test"])
    categories = {item.canonical: item.category_id for item in stable_id_map()}
    examples: list[TestExample] = []
    clause_ids: set[str] = set()
    with (data / "clauses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("contract_id") not in seen:
                raise ValueError("clause has no valid split assignment")
            if row["contract_id"] not in test_ids:
                continue
            if not all(
                isinstance(row.get(key), str)
                for key in ("clause_id", "text", "category")
            ):
                raise ValueError("invalid held-out clause")
            if row["clause_id"] in clause_ids or row["category"] not in categories:
                raise ValueError("duplicate clause ID or unknown taxonomy")
            clause_ids.add(row["clause_id"])
            examples.append(
                TestExample(
                    row["clause_id"],
                    row["contract_id"],
                    row["text"],
                    categories[row["category"]],
                )
            )
    if not examples:
        raise ValueError("empty held-out split")
    examples.sort(key=lambda item: item.clause_id)
    payload = [
        [item.clause_id, item.contract_id, item.text, item.category_id]
        for item in examples
    ]
    checksum = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return examples, checksum, sha256_file(data / "splits.json")


def _provider(
    manifest: ArtifactManifest,
    manifest_path: Path,
    max_sequence_length: int,
    max_new_tokens: int,
) -> ClauseClassifierProvider:
    assert manifest.adapter_path is not None
    return LocalTransformerProvider(
        Path(manifest.base_model),
        (manifest_path.parent / manifest.adapter_path).resolve(),
        None,
        "cuda",
        max_sequence_length,
        target_representation=manifest.target_representation,
        target_representation_version=manifest.target_representation_version,
        prompt_template_version=manifest.prompt_version,
        base_revision=manifest.base_revision,
        max_new_tokens=max_new_tokens,
        artifact_id=manifest.artifact_id,
        checkpoint_step=manifest.checkpoint_step,
        candidate_status=manifest.release_status,
    )


async def evaluate_locked_test(
    lock_path: Path,
    manifest_path: Path,
    data: Path,
    output: Path,
    predictions: Path,
    *,
    max_sequence_length: int = 1024,
    max_new_tokens: int = 24,
) -> dict[str, object]:
    lock, manifest = verify_authorization(lock_path, manifest_path)
    if max_sequence_length <= 0 or max_new_tokens <= 0:
        raise ValueError("generation limits must be positive")
    receipt = lock_path.with_name(lock_path.name + ".held-out-attempt.json")
    paths = [
        p.resolve() for p in (output, predictions, receipt, lock_path, manifest_path)
    ]
    if len(set(paths)) != len(paths) or any(
        p.exists() for p in (output, predictions, receipt)
    ):
        raise ValueError("held-out evidence/attempt already exists or paths conflict")
    provider = _provider(manifest, manifest_path, max_sequence_length, max_new_tokens)
    try:
        ready, _ = provider.is_ready()
        if not ready or provider.is_mock or provider.provider_type != "transformer":
            raise ValueError(
                "locked real transformer is not ready; no fallback permitted"
            )
        # Recheck after potentially long model loading, before any data access.
        verify_authorization(lock_path, manifest_path)
        timestamp = datetime.now(UTC).isoformat()
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "artifact_id": manifest.artifact_id,
                    "checkpoint_step": manifest.checkpoint_step,
                    "manifest_checksum": lock.manifest_checksum,
                    "started_at": timestamp,
                    "label": "HELD-OUT ATTEMPT — DO NOT DELETE OR RETRY",
                },
                stream,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        examples, checksum, split_checksum = _load_test_only(data)
        categories = {item.category_id: item for item in stable_id_map()}
        labels = sorted(categories)
        targets: list[str] = []
        predicted: list[str] = []
        statuses: Counter[str] = Counter()
        with predictions.open("x", encoding="utf-8") as stream:
            for item in examples:
                result = await provider.classify(item.text)
                status, reason, correct = diagnose_id_output(
                    result.raw_output, categories[item.category_id]
                )
                accepted = (
                    result.raw_output.strip() if status == "exact" else "__invalid__"
                )
                targets.append(item.category_id)
                predicted.append(accepted)
                statuses[status] += 1
                stream.write(
                    json.dumps(
                        {
                            "split": "test",
                            "clause_id": item.clause_id,
                            "canonical_target_id": item.category_id,
                            "raw_generated_text": result.raw_output,
                            "normalized_generated_text": result.raw_output.strip(),
                            "validation_status": status,
                            "status_reason": reason,
                            "exact_match": correct,
                            "generated_token_count": None,
                            "target_token_count": None,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        total = len(examples)
        report: dict[str, object] = {
            **classification_metrics(targets, predicted, labels),
            "schema_version": REPORT_SCHEMA,
            "label": "HELD-OUT FINAL TEST",
            "split": "test",
            "artifact_id": manifest.artifact_id,
            "checkpoint_step": manifest.checkpoint_step,
            "experiment_id": manifest.adapter_experiment_id,
            "adapter_checksum": manifest.adapter_checksum,
            "manifest_checksum": lock.manifest_checksum,
            "selection_checksum": lock.selection_checksum,
            "config_checksum": manifest.config_checksum,
            "training_commit": manifest.training_commit,
            "authorization_lock_checksum": sha256_file(lock_path),
            "attempt_receipt_checksum": sha256_file(receipt),
            "base_model": manifest.base_model,
            "base_revision": manifest.base_revision,
            "prompt_version": manifest.prompt_version,
            "target_representation": manifest.target_representation,
            "target_representation_version": manifest.target_representation_version,
            "taxonomy_version": manifest.taxonomy_version,
            "stable_id_map_checksum": manifest.stable_id_map_checksum,
            "test_dataset_checksum": checksum,
            "split_assignments_checksum": split_checksum,
            "total_examples": total,
            "category_coverage": {
                "categories_present": len(set(targets)),
                "total_supported_categories": len(labels),
                "examples_per_category": dict(Counter(targets)),
            },
            "exact_id_count": statuses["exact"],
            "exact_id_output_rate": statuses["exact"] / total,
            "invalid_output_count": total - statuses["exact"],
            "invalid_output_rate": 1 - statuses["exact"] / total,
            "malformed_output_count": statuses["malformed"],
            "empty_output_count": statuses["empty"],
            "prediction_distribution": dict(Counter(predicted)),
            "test_loss": None,
            "generated_token_diagnostics": None,
            "limitations": [
                "Production provider does not expose token counts "
                "or teacher-forced loss"
            ],
            "generation": {
                "max_new_tokens": max_new_tokens,
                "max_sequence_length": max_sequence_length,
                "do_sample": False,
            },
            "provider": provider.name,
            "is_mock": False,
            "timestamp": timestamp,
            "test_evaluated": True,
            "incomplete_training_selection_reason": (
                lock.incomplete_training_selection_reason
            ),
            "predictions_checksum": sha256_file(predictions),
        }
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        await provider.close()
