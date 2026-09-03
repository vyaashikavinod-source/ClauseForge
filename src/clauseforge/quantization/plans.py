"""Deterministic AWQ and llama.cpp command plans without execution."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

from clauseforge.quantization.config import QuantizationConfig


class OptionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CommandPlan:
    stage: str
    argv: tuple[str, ...]
    executed: bool = False


def require_awq() -> None:
    if importlib.util.find_spec("awq") is None:
        raise OptionalDependencyError(
            "AWQ execution requires optional AutoAWQ and a suitable GPU"
        )


def awq_plan(
    config: QuantizationConfig,
    *,
    parent_artifact_id: str | None = None,
    parent_artifact_checksum: str | None = None,
) -> dict[str, object]:
    config.validate()
    if config.method != "awq":
        raise ValueError("AWQ plan requires method=awq")
    return {
        "stage": "quantize_awq",
        "configuration": config.to_dict(),
        "executed": False,
        "requirements": ["final merged model", "AutoAWQ", "CUDA GPU"],
        "parent_artifact_id": parent_artifact_id,
        "parent_artifact_checksum": parent_artifact_checksum,
    }


def gguf_plan(config: QuantizationConfig) -> tuple[CommandPlan, CommandPlan]:
    config.validate()
    if config.method != "gguf" or config.llama_cpp_path is None:
        raise ValueError("GGUF plan requires method=gguf")
    intermediate = config.output_path.with_suffix(".f16.gguf")
    return (
        CommandPlan(
            "convert_f16",
            (
                "python",
                str(config.llama_cpp_path / "convert_hf_to_gguf.py"),
                str(config.merged_model_path),
                "--outfile",
                str(intermediate),
                "--outtype",
                "f16",
            ),
        ),
        CommandPlan(
            "quantize",
            (
                str(config.llama_cpp_path / "llama-quantize"),
                str(intermediate),
                str(config.output_path),
                config.quantization_type,
            ),
        ),
    )


def serialize_plans(plans: tuple[CommandPlan, ...]) -> list[dict[str, object]]:
    return [asdict(plan) for plan in plans]
