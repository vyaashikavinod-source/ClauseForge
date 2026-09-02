"""Historical checkpoint compatibility for validation-only diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import cast

from clauseforge.training.config import TrainingConfig
from clauseforge.training.targets import (
    CANONICAL_TARGET_VERSION,
    stable_id_map_checksum,
)


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    compatible: bool
    model_critical_matches: dict[str, bool]
    training_critical_matches: dict[str, bool]
    evaluation_overrides: dict[str, object]
    ignored_nonstructural_differences: tuple[str, ...]
    blocking_differences: tuple[str, ...]
    historical_training_subset: dict[str, object]
    historical_validation_subset: dict[str, object]
    evaluation_validation_subset: dict[str, object]
    training_subset_match: bool
    validation_subset_match: bool
    evaluation_override: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def historical_experiment_id(config: dict[str, object]) -> str:
    """Reproduce the identity from the persisted configuration as written."""
    model = cast(dict[str, object], config["model"])
    lora = cast(dict[str, object], config["lora"])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    short_name = str(model["name"]).rsplit("/", 1)[-1].lower().replace("_", "-")
    rank = int(cast(int, lora["rank"]))
    seed = int(cast(int, config["seed"]))
    return f"{short_name}_lora-r{rank}_seed{seed}_{digest}"


def _nested(config: dict[str, object], section: str, field: str) -> object:
    return cast(dict[str, object], config[section]).get(field)


def checkpoint_evaluation_compatibility(
    current: TrainingConfig,
    historical: dict[str, object],
    pilot: dict[str, object],
    resume: dict[str, object],
    adapter: dict[str, object],
    *,
    expected_run_mode: str,
    expected_subset_checksum: str,
    historical_taxonomy: list[str],
    current_taxonomy: tuple[str, ...],
    historical_training_subset: dict[str, object],
    historical_validation_subset: dict[str, object],
    evaluation_validation_subset: dict[str, object],
    evaluation_override: bool,
) -> CompatibilityReport:
    """Compare structural lineage while allowing explicit evaluation controls."""
    current_dict = current.to_dict()
    model_fields = (
        "name",
        "revision",
        "tokenizer_name",
        "family",
        "quantization",
        "quant_type",
        "double_quant",
        "precision",
    )
    model_matches = {
        f"model.{field}": _nested(historical, "model", field)
        == _nested(current_dict, "model", field)
        for field in model_fields
    }
    historical_representation = str(
        historical.get("target_representation", "canonical_question")
    )
    historical_target_version = str(
        historical.get("target_representation_version", CANONICAL_TARGET_VERSION)
    )
    model_matches["target_representation"] = (
        historical_representation == current.target_representation
    )
    model_matches["target_representation_version"] = (
        historical_target_version == current.target_representation_version
    )
    for field in ("rank", "alpha", "target_modules"):
        historical_value = _nested(historical, "lora", field)
        current_value = _nested(current_dict, "lora", field)
        if field == "target_modules":
            historical_value = tuple(cast(list[str], historical_value))
            current_value = tuple(cast(tuple[str, ...], current_value))
        model_matches[f"lora.{field}"] = historical_value == current_value
    model_matches["adapter.rank"] = adapter.get("rank") == _nested(
        historical, "lora", "rank"
    )
    model_matches["adapter.alpha"] = adapter.get("alpha") == _nested(
        historical, "lora", "alpha"
    )
    model_matches["adapter.target_modules"] = tuple(
        cast(list[str], adapter.get("target_modules", []))
    ) == tuple(cast(list[str], _nested(historical, "lora", "target_modules")))
    adapter_representation = str(
        adapter.get("target_representation", "canonical_question")
    )
    adapter_version = str(
        adapter.get("target_representation_version", CANONICAL_TARGET_VERSION)
    )
    model_matches["adapter.target_representation"] = (
        adapter_representation == historical_representation
    )
    model_matches["adapter.target_representation_version"] = (
        adapter_version == historical_target_version
    )
    if historical_representation == "category_id":
        model_matches["stable_id_map_checksum"] = (
            adapter.get("stable_id_map_checksum") == stable_id_map_checksum()
        )

    stored_id = str(resume.get("experiment_id", ""))
    combined_matches = (
        resume.get("subset_checksum")
        == expected_subset_checksum
        == pilot.get("selection_checksum")
    )
    training_subset_match = combined_matches and pilot.get(
        "train_examples"
    ) == historical_training_subset.get("selected_count")
    validation_subset_match = combined_matches and pilot.get(
        "validation_examples"
    ) == historical_validation_subset.get("selected_count")
    training_matches = {
        "experiment_identity": stored_id == historical_experiment_id(historical),
        "run_mode": resume.get("run_mode") == expected_run_mode,
        "selected_subset_checksum": (
            training_subset_match and validation_subset_match
            if expected_run_mode == "pilot"
            else resume.get("subset_checksum") == expected_subset_checksum == ""
        ),
        "prompt_template_version": historical.get("prompt_template_version")
        == current.prompt_template_version,
        "max_sequence_length": _nested(historical, "data", "max_sequence_length")
        == current.data.max_sequence_length,
        "taxonomy": historical_taxonomy == list(current_taxonomy),
    }
    historical_generation = _nested(historical, "data", "validation_max_new_tokens")
    ignored: list[str] = []
    if historical_generation != current.data.validation_max_new_tokens:
        ignored.append("data.validation_max_new_tokens")
    blocking = tuple(
        name
        for group in (model_matches, training_matches)
        for name, matches in group.items()
        if not matches
    )
    return CompatibilityReport(
        compatible=not blocking,
        model_critical_matches=model_matches,
        training_critical_matches=training_matches,
        evaluation_overrides={
            "validation_max_new_tokens": current.data.validation_max_new_tokens,
            "checkpoint_validation_max_new_tokens": historical_generation,
        },
        ignored_nonstructural_differences=tuple(ignored),
        blocking_differences=blocking,
        historical_training_subset=historical_training_subset,
        historical_validation_subset=historical_validation_subset,
        evaluation_validation_subset=evaluation_validation_subset,
        training_subset_match=training_subset_match,
        validation_subset_match=validation_subset_match,
        evaluation_override=evaluation_override,
    )
