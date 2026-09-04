from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest
import yaml
from scripts.smoke_release_http import run

from clauseforge.config import Settings
from clauseforge.release.handoff import environment_report
from clauseforge.release.readiness import assess_readiness


class Response(io.BytesIO):
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        super().__init__(json.dumps(payload).encode())
        self.status = status
        self.headers = {
            "X-Request-ID": "synthetic",
            "X-Content-Type-Options": "nosniff",
        }


@pytest.mark.parametrize("mock", [True, False])
def test_http_smoke_checks_real_identity_and_operations(
    monkeypatch: pytest.MonkeyPatch, mock: bool
) -> None:
    calls = 0

    def request(req: urllib.request.Request, timeout: int) -> Response:
        nonlocal calls
        if req.full_url.endswith("/ready"):
            return Response({"artifact_id": "selected", "checkpoint_step": 800})
        if req.full_url.endswith("/v1/classify"):
            calls += 1
            return Response(
                {
                    "provider": "active-trained-candidate",
                    "processing": {"is_mock": mock, "cache_hit": calls == 2},
                },
                429 if calls > 2 else 200,
            )
        return Response({})

    monkeypatch.setattr("urllib.request.urlopen", request)
    if mock:
        with pytest.raises(ValueError, match="real classification"):
            run("http://fixture", "selected", 800)
    else:
        result = run("http://fixture", "selected", 800)
        assert result["passed"] is True
        assert result["logs_verified"] is False


def test_release_container_configuration_is_explicit() -> None:
    compose = yaml.safe_load(
        Path("docker-compose.release.yml").read_text(encoding="utf-8")
    )
    api = compose["services"]["api"]
    assert api["gpus"] == "all"
    assert api["environment"]["CLAUSEFORGE_MODEL_BACKEND"] == "transformer"
    assert api["read_only"] is True
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    dockerfile = Path("Dockerfile.gpu").read_text(encoding="utf-8")
    assert 'test -n "$BNB_VERSION"' in dockerfile
    assert "USER clauseforge" in dockerfile
    default = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert (
        default["services"]["api"]["environment"]["CLAUSEFORGE_MODEL_BACKEND"] == "mock"
    )


def test_environment_has_no_fabricated_gpu_results() -> None:
    report = environment_report()
    assert report["gpu_results"] is None
    assert report["python"]


def test_manifest_readiness_uses_actual_gates() -> None:
    report = assess_readiness(
        Settings(
            model_artifact_manifest=Path(
                "configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json"
            )
        )
    )
    checks = {check.name: check for check in report.model_artifact}
    assert "rank_sweep" not in checks
    assert checks["held_out_test"].status == "blocked"
    assert report.overall_status == "blocked"
