from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clauseforge.artifacts.comparison import compare_candidates
from clauseforge.artifacts.registry import ArtifactRegistry
from clauseforge.artifacts.validation import (
    load_manifest,
    write_manifest,
)
from clauseforge.config import Settings
from clauseforge.serving.app import create_app
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.serving.providers.local_transformer import LocalTransformerProvider

RC0 = Path("configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json")


def test_candidate_comparison_uses_ordered_validation_rules(tmp_path: Path) -> None:
    base = load_manifest(RC0)
    assert base.validation_summary is not None
    first = replace(base, artifact_id="step-700", checkpoint_step=700)
    better_invalid = replace(
        base,
        artifact_id="step-1400",
        checkpoint_step=1400,
        validation_summary=replace(
            base.validation_summary,
            invalid_output_rate=base.validation_summary.invalid_output_rate - 0.01,
        ),
    )
    first_path, second_path = tmp_path / "700.json", tmp_path / "1400.json"
    write_manifest(first_path, first)
    write_manifest(second_path, better_invalid)
    report = compare_candidates(first_path, second_path)
    assert report.recommended_artifact_id == "step-1400"
    assert report.reason == "invalid_output_rate"
    assert report.split == "validation" and not report.test_evaluated
    earlier = replace(better_invalid, validation_summary=base.validation_summary)
    write_manifest(second_path, earlier)
    assert (
        compare_candidates(first_path, second_path).recommended_artifact_id
        == "step-700"
    )


def test_candidate_comparison_rejects_test_evidence(tmp_path: Path) -> None:
    candidate = replace(load_manifest(RC0), test_evaluated=True)
    path = tmp_path / "candidate.json"
    write_manifest(path, candidate)
    with pytest.raises(ValueError, match="held-out test"):
        compare_candidates(path, RC0)


def test_real_mode_is_fail_closed_without_cuda_or_adapter() -> None:
    provider = LocalTransformerProvider(
        Path("Qwen/Qwen2.5-7B-Instruct"), None, None, "cpu", 1024
    )
    assert provider.is_mock is False
    assert provider.is_ready()[0] is False


class ActiveCandidateProvider:
    name = "active-trained-candidate"
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    provider_type = "transformer"
    is_mock = False
    artifact_id = "clauseforge-qwen25-7b-r8-step700"
    checkpoint_step = 700
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    candidate_status = "release_candidate"
    target_representation = "category_id"
    target_representation_version = "cuad-category-id-v1"

    def is_ready(self) -> tuple[bool, str | None]:
        return True, None

    async def classify(self, text: str) -> ProviderResult:
        return ProviderResult("governing_law", "governing_law", None)

    async def close(self) -> None:
        return None


def test_active_candidate_readiness_and_manifest_driven_identity() -> None:
    settings = Settings(
        environment="test",
        model_provider="real",
        model_artifact_manifest=Path("candidate.json"),
    )
    with TestClient(create_app(settings, provider=ActiveCandidateProvider())) as client:
        ready = client.get("/ready")
        classified = client.post("/v1/classify", json={"text": "governed by law"})
    assert ready.status_code == 200
    assert ready.json()["artifact_id"] == "clauseforge-qwen25-7b-r8-step700"
    assert ready.json()["checkpoint_step"] == 700
    assert ready.json()["candidate_status"] == "release_candidate"
    assert classified.status_code == 200


def test_activation_preserves_previous_candidate(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "registry")
    base = load_manifest(RC0)
    for artifact_id, step in (
        ("step-700", 700),
        ("step-1400", 1400),
        ("step-2100", 2100),
    ):
        path = tmp_path / f"{artifact_id}.json"
        write_manifest(
            path, replace(base, artifact_id=artifact_id, checkpoint_step=step)
        )
        registry.register(path)
        registry.activate(artifact_id)
    active = registry.read_active()
    assert active.current_artifact_id == "step-2100"
    assert active.previous_artifact_id == "step-1400"
    assert registry.rollback().current_artifact_id == "step-1400"
