from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from clauseforge.quantization.config import QuantizationConfigError, load_config
from clauseforge.quantization.manifest import DeploymentManifest, QuantizationManifest
from clauseforge.quantization.merge import (
    AdapterMetadata,
    ModelIdentity,
    merge_plan,
    validate_merge,
)
from clauseforge.quantization.plans import (
    OptionalDependencyError,
    awq_plan,
    gguf_plan,
    require_awq,
)
from clauseforge.quantization.validation import validate_deployment
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION


def test_config_parsing_and_invalid_combinations() -> None:
    awq = load_config(Path("quantization/configs/qwen25_7b_awq.yaml"))
    gguf = load_config(Path("quantization/configs/qwen25_7b_gguf_q4km.yaml"))
    assert awq_plan(awq)["executed"] is False
    assert gguf_plan(gguf)[1].argv[-1] == "Q4_K_M"
    with pytest.raises(QuantizationConfigError, match="AWQ requires"):
        replace(awq, bits=8).validate()
    with pytest.raises(QuantizationConfigError, match="incompatible"):
        replace(gguf, bits=5).validate()


def test_awq_dependency_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(OptionalDependencyError, match="AutoAWQ"):
        require_awq()


def test_merge_mismatch_and_plan() -> None:
    base = ModelIdentity("Qwen", "abc", "Qwen2ForCausalLM")
    adapter = AdapterMetadata(
        "Qwen",
        "abc",
        "Qwen2ForCausalLM",
        "exp",
        "rev",
        TAXONOMY_VERSION,
        PROMPT_TEMPLATE_VERSION,
        8,
        ("q_proj",),
    )
    assert merge_plan(base, adapter, "out", "commit")["executed"] is False
    with pytest.raises(ValueError, match="base_revision"):
        validate_merge(base, replace(adapter, base_revision="wrong"))


def test_manifests_and_release_checksum(tmp_path: Path) -> None:
    artifact = QuantizationManifest(
        "a",
        "Qwen",
        "abc",
        "exp",
        "rev",
        "sum",
        TAXONOMY_VERSION,
        PROMPT_TEMPLATE_VERSION,
        "commit",
        "gguf",
        {"type": "Q4_K_M"},
        "gguf",
        1024,
        "now",
        "prepared",
        ("not executed",),
    )
    assert artifact.to_dict()["artifact_id"] == "a"
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    checksum = hashlib.sha256(b"fixture").hexdigest()
    manifest = DeploymentManifest(
        "d",
        "Qwen",
        "abc",
        "exp",
        "a",
        "llamacpp",
        TAXONOMY_VERSION,
        PROMPT_TEMPLATE_VERSION,
        1024,
        160,
        "commit",
        "test",
        "now",
        "prepared",
        ("fixture",),
        {"model.gguf": checksum},
    )
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    assert validate_deployment(path).deployment_id == "d"
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_deployment(path)
