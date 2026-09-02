"""Factual offline release-readiness assessment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from clauseforge import __version__
from clauseforge.config import Settings
from clauseforge.quantization.validation import validate_deployment
from clauseforge.serving.constants import CUAD_TAXONOMY, TAXONOMY_VERSION
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION

Status = Literal["ready", "blocked", "not_applicable", "pending"]


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    application_version: str
    infrastructure: tuple[ReadinessCheck, ...]
    model_artifact: tuple[ReadinessCheck, ...]
    deployment: tuple[ReadinessCheck, ...]
    production_validation: tuple[ReadinessCheck, ...]
    overall_status: Status

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_readiness(
    settings: Settings,
    *,
    deployment_manifest: Path | None = None,
    quantization_manifest: Path | None = None,
) -> ReadinessReport:
    settings.validate()
    backend_issues = settings.backend_configuration_issues()
    infrastructure = (
        ReadinessCheck("application_metadata", "ready", f"version {__version__}"),
        ReadinessCheck(
            "taxonomy",
            "ready" if len(CUAD_TAXONOMY) == 41 else "blocked",
            TAXONOMY_VERSION,
        ),
        ReadinessCheck("prompt_template", "ready", PROMPT_TEMPLATE_VERSION),
        ReadinessCheck(
            "serving_configuration",
            "blocked" if backend_issues else "ready",
            "; ".join(backend_issues) or settings.model_provider,
        ),
    )
    artifact_configured = any(
        value is not None
        for value in (settings.model_path, settings.adapter_path, settings.gguf_path)
    )
    model = (
        ReadinessCheck("final_adapter_selected", "blocked", "rank sweep incomplete"),
        ReadinessCheck(
            "model_artifact",
            "pending" if artifact_configured else "blocked",
            "final adapter artifact is not available",
        ),
        ReadinessCheck(
            "quantization_executed",
            _manifest_status(quantization_manifest),
            "manifest supplied for offline review"
            if quantization_manifest
            else "actual AWQ/GGUF quantization has not been validated",
        ),
    )
    deployment = (
        ReadinessCheck(
            "deployment_manifest",
            _deployment_manifest_status(deployment_manifest),
            "manifest supplied for offline review"
            if deployment_manifest
            else "final deployment manifest is not available",
        ),
        ReadinessCheck("container_definition", "ready", "CPU mock image supported"),
        ReadinessCheck("final_serving_benchmark", "blocked", "not executed"),
    )
    validation = (
        ReadinessCheck("rank_sweep", "blocked", "not complete"),
        ReadinessCheck("held_out_test", "blocked", "sealed and not evaluated"),
        ReadinessCheck("final_model_safety", "blocked", "not rerun"),
        ReadinessCheck("final_model_ood", "blocked", "not run"),
    )
    return ReadinessReport(
        __version__, infrastructure, model, deployment, validation, "blocked"
    )


def serialize_report(report: ReadinessReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _manifest_status(path: Path | None) -> Status:
    if path is None:
        return "blocked"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "blocked"
    return "pending" if isinstance(value, dict) else "blocked"


def _deployment_manifest_status(path: Path | None) -> Status:
    if path is None:
        return "blocked"
    try:
        validate_deployment(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "blocked"
    return "pending"
