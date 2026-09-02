"""Typed, validated experiment configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

from clauseforge.training.targets import (
    CANONICAL_TARGET_VERSION,
    TargetRepresentation,
    validate_target_version,
)

Precision = Literal["float32", "float16", "bfloat16"]
Quantization = Literal["none", "4bit"]
OptimizerName = Literal["adamw_torch", "paged_adamw_8bit"]
SchedulerName = Literal["linear", "cosine"]


class ConfigurationError(ValueError):
    """Raised before an unsafe or incoherent experiment can start."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    revision: str
    tokenizer_name: str
    family: str
    license: str
    precision: Precision = "float32"
    quantization: Quantization = "none"
    quant_type: str = "nf4"
    double_quant: bool = True
    use_cache: bool = False


@dataclass(frozen=True, slots=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 1
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    optimizer: OptimizerName = "adamw_torch"
    scheduler: SchedulerName = "linear"
    gradient_checkpointing: bool = False
    gradient_checkpointing_use_reentrant: bool = False
    save_steps: int = 100
    eval_steps: int = 100


@dataclass(frozen=True, slots=True)
class DataConfig:
    max_sequence_length: int = 1024
    validation_max_new_tokens: int = 160
    max_train_examples: int | None = None
    max_validation_examples: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: ModelConfig
    lora: LoraSettings
    optimization: OptimizationConfig
    data: DataConfig
    seed: int
    output_dir: Path
    prompt_template_version: str = "cuad-classification-v1"
    target_representation: TargetRepresentation = "canonical_question"
    target_representation_version: str = CANONICAL_TARGET_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.model.name.strip() or not self.model.revision.strip():
            raise ConfigurationError("model name and immutable revision are required")
        if self.model.license.lower() in {"", "unknown", "unchecked"}:
            raise ConfigurationError("reviewed model license metadata is required")
        if self.model.precision not in {"float32", "float16", "bfloat16"}:
            raise ConfigurationError(f"unsupported precision: {self.model.precision}")
        if self.model.quantization not in {"none", "4bit"}:
            raise ConfigurationError(
                f"unsupported quantization: {self.model.quantization}"
            )
        if self.data.max_sequence_length < 32:
            raise ConfigurationError("max_sequence_length must be at least 32")
        if self.data.validation_max_new_tokens <= 0:
            raise ConfigurationError("validation_max_new_tokens must be positive")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ConfigurationError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora.dropout < 1.0:
            raise ConfigurationError("LoRA dropout must be in [0, 1)")
        if self.optimization.learning_rate <= 0:
            raise ConfigurationError("learning rate must be positive")
        if (
            min(
                self.optimization.epochs,
                self.optimization.batch_size,
                self.optimization.gradient_accumulation,
            )
            <= 0
        ):
            raise ConfigurationError("epochs and batch sizes must be positive")
        if not 0.0 <= self.optimization.warmup_ratio < 1.0:
            raise ConfigurationError("warmup_ratio must be in [0, 1)")
        if self.model.quantization == "4bit" and self.model.precision == "float32":
            raise ConfigurationError("4-bit training requires float16 or bfloat16")
        if self.model.quant_type != "nf4":
            raise ConfigurationError("Phase 3B supports only NF4 quantization")
        if self.optimization.optimizer not in {"adamw_torch", "paged_adamw_8bit"}:
            raise ConfigurationError("unsupported optimizer")
        if self.optimization.scheduler not in {"linear", "cosine"}:
            raise ConfigurationError("unsupported scheduler")
        if min(self.optimization.save_steps, self.optimization.eval_steps) <= 0:
            raise ConfigurationError("save_steps and eval_steps must be positive")
        if self.output_dir.resolve() == Path.cwd().resolve():
            raise ConfigurationError("output_dir may not be the repository root")
        try:
            validate_target_version(
                self.target_representation,
                self.target_representation_version,
                self.prompt_template_version,
            )
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        return value

    @property
    def experiment_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        short_name = self.model.name.rsplit("/", 1)[-1].lower().replace("_", "-")
        return f"{short_name}_lora-r{self.lora.rank}_seed{self.seed}_{digest}"


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _str(mapping: dict[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string")
    return value


def _int(mapping: dict[str, object], key: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer")
    return value


def _float(mapping: dict[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be numeric")
    return float(value)


def _optional_int(mapping: dict[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer or null")
    return value


def load_config(path: Path) -> TrainingConfig:
    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
        model = _mapping(root["model"], "model")
        lora = _mapping(root["lora"], "lora")
        optimization = _mapping(root["optimization"], "optimization")
        data = _mapping(root["data"], "data")
        config = TrainingConfig(
            model=ModelConfig(
                name=_str(model, "name"),
                revision=_str(model, "revision"),
                tokenizer_name=_str(model, "tokenizer_name"),
                family=_str(model, "family"),
                license=_str(model, "license"),
                precision=cast(Precision, model.get("precision", "float32")),
                quantization=cast(Quantization, model.get("quantization", "none")),
                quant_type=str(model.get("quant_type", "nf4")),
                double_quant=bool(model.get("double_quant", True)),
                use_cache=bool(model.get("use_cache", False)),
            ),
            lora=LoraSettings(
                rank=_int(lora, "rank", 16),
                alpha=_int(lora, "alpha", 32),
                dropout=_float(lora, "dropout", 0.05),
                target_modules=tuple(cast(list[str], lora.get("target_modules", []))),
            ),
            optimization=OptimizationConfig(
                learning_rate=_float(optimization, "learning_rate", 2e-4),
                epochs=_int(optimization, "epochs", 1),
                batch_size=_int(optimization, "batch_size", 1),
                gradient_accumulation=_int(optimization, "gradient_accumulation", 1),
                warmup_ratio=_float(optimization, "warmup_ratio", 0.0),
                weight_decay=_float(optimization, "weight_decay", 0.0),
                optimizer=cast(
                    OptimizerName, optimization.get("optimizer", "adamw_torch")
                ),
                scheduler=cast(SchedulerName, optimization.get("scheduler", "linear")),
                gradient_checkpointing=bool(
                    optimization.get("gradient_checkpointing", False)
                ),
                gradient_checkpointing_use_reentrant=bool(
                    optimization.get("gradient_checkpointing_use_reentrant", False)
                ),
                save_steps=_int(optimization, "save_steps", 100),
                eval_steps=_int(optimization, "eval_steps", 100),
            ),
            data=DataConfig(
                max_sequence_length=_int(data, "max_sequence_length", 1024),
                validation_max_new_tokens=_int(data, "validation_max_new_tokens", 160),
                max_train_examples=_optional_int(data, "max_train_examples"),
                max_validation_examples=_optional_int(data, "max_validation_examples"),
            ),
            seed=int(cast(int, root["seed"])),
            output_dir=Path(cast(str, root["output_dir"])),
            prompt_template_version=str(
                root.get("prompt_template_version", "cuad-classification-v1")
            ),
            target_representation=cast(
                TargetRepresentation,
                root.get("target_representation", "canonical_question"),
            ),
            target_representation_version=str(
                root.get("target_representation_version", CANONICAL_TARGET_VERSION)
            ),
            tags=tuple(cast(list[str], root.get("tags", []))),
        )
    except (KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid configuration at {path}: {exc}") from exc
    config.validate()
    return config
