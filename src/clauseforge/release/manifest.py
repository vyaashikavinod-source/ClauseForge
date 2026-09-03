"""Strict final release manifest with unavailable evidence left null."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FinalReleaseManifest:
    schema_version: str
    release_id: str
    artifact_id: str
    git_commit: str
    base_model: str
    base_revision: str
    taxonomy_version: str
    prompt_version: str
    target_version: str
    test_metrics: dict[str, object] | None
    safety_summary: dict[str, object] | None
    ood_summary: dict[str, object] | None
    quantization: dict[str, object] | None
    benchmark_summary: dict[str, object] | None
    deployment_bundle_checksum: str | None
    release_timestamp: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self, *, final: bool = False) -> None:
        if self.schema_version != "clauseforge-final-release-v1":
            raise ValueError("unsupported final release schema")
        required = (
            self.release_id,
            self.artifact_id,
            self.git_commit,
            self.base_model,
            self.base_revision,
            self.taxonomy_version,
            self.prompt_version,
            self.target_version,
        )
        if not all(required):
            raise ValueError("final release identity is incomplete")
        evidence = (
            self.test_metrics,
            self.safety_summary,
            self.ood_summary,
            self.quantization,
            self.benchmark_summary,
            self.deployment_bundle_checksum,
            self.release_timestamp,
        )
        if final and not all(item is not None for item in evidence):
            raise ValueError("final release evidence is incomplete")
