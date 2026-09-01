"""PEFT adapter/base compatibility validation and explicit merge planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    base_model: str
    base_revision: str
    architecture: str


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    base_model: str
    base_revision: str
    architecture: str
    experiment_id: str
    adapter_revision: str
    taxonomy_version: str
    prompt_version: str
    lora_rank: int
    target_modules: tuple[str, ...]


def validate_merge(base: ModelIdentity, adapter: AdapterMetadata) -> None:
    mismatches = []
    for key in ("base_model", "base_revision", "architecture"):
        if getattr(base, key) != getattr(adapter, key):
            mismatches.append(key)
    if (
        not adapter.experiment_id
        or not adapter.adapter_revision
        or adapter.lora_rank <= 0
    ):
        mismatches.append("adapter_metadata")
    if (
        not adapter.taxonomy_version
        or not adapter.prompt_version
        or not adapter.target_modules
    ):
        mismatches.append("training_contract")
    if mismatches:
        raise ValueError(f"base/adapter incompatibility: {', '.join(mismatches)}")


def merge_plan(
    base: ModelIdentity, adapter: AdapterMetadata, output: str, build_commit: str
) -> dict[str, object]:
    validate_merge(base, adapter)
    return {
        "operation": "merge_peft_adapter",
        "base": asdict(base),
        "adapter": asdict(adapter),
        "output": output,
        "build_commit": build_commit,
        "executed": False,
    }
