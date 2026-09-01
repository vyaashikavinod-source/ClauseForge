"""Typed method-specific quantization configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

Method = Literal["awq", "gguf"]
QuantType = Literal["awq", "Q4_K_M", "Q5_K_M", "Q8_0"]


class QuantizationConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    method: Method
    source_model: str
    source_revision: str
    adapter_path: Path
    adapter_metadata: Path
    merged_model_path: Path
    output_path: Path
    bits: int
    group_size: int | None
    quantization_type: QuantType
    calibration_source: Path | None
    max_sequence_length: int
    device: str
    seed: int
    llama_cpp_path: Path | None = None

    def validate(self) -> None:
        if not self.source_model or not self.source_revision:
            raise QuantizationConfigError(
                "source model and immutable revision are required"
            )
        if self.output_path == self.merged_model_path:
            raise QuantizationConfigError("output must not overwrite merged model")
        if self.max_sequence_length < 32:
            raise QuantizationConfigError("max sequence length must be at least 32")
        if self.method == "awq":
            if (self.bits, self.group_size, self.quantization_type) != (4, 128, "awq"):
                raise QuantizationConfigError(
                    "AWQ requires bits=4, group_size=128, type=awq"
                )
            if self.calibration_source is None or self.llama_cpp_path is not None:
                raise QuantizationConfigError(
                    "AWQ requires calibration and rejects llama.cpp settings"
                )
        elif self.method == "gguf":
            expected = {"Q4_K_M": 4, "Q5_K_M": 5, "Q8_0": 8}.get(self.quantization_type)
            if expected is None or self.bits != expected:
                raise QuantizationConfigError(
                    "GGUF quantization type and bits are incompatible"
                )
            if (
                self.group_size is not None
                or self.calibration_source is not None
                or self.llama_cpp_path is None
            ):
                raise QuantizationConfigError(
                    "GGUF requires llama.cpp and rejects AWQ-only settings"
                )
        else:
            raise QuantizationConfigError("method must be awq or gguf")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "adapter_path",
            "adapter_metadata",
            "merged_model_path",
            "output_path",
            "calibration_source",
            "llama_cpp_path",
        ):
            value[key] = str(value[key]) if value[key] is not None else None
        return value


def _path(value: object, key: str, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise QuantizationConfigError(f"{key} must be a path")
    return Path(value)


def load_config(path: Path) -> QuantizationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise QuantizationConfigError("configuration must be a mapping")
        config = QuantizationConfig(
            method=cast(Method, raw["method"]),
            source_model=str(raw["source_model"]),
            source_revision=str(raw["source_revision"]),
            adapter_path=cast(Path, _path(raw["adapter_path"], "adapter_path")),
            adapter_metadata=cast(
                Path, _path(raw["adapter_metadata"], "adapter_metadata")
            ),
            merged_model_path=cast(
                Path, _path(raw["merged_model_path"], "merged_model_path")
            ),
            output_path=cast(Path, _path(raw["output_path"], "output_path")),
            bits=int(raw["bits"]),
            group_size=cast(int | None, raw.get("group_size")),
            quantization_type=cast(QuantType, raw["quantization_type"]),
            calibration_source=_path(
                raw.get("calibration_source"), "calibration_source", True
            ),
            max_sequence_length=int(raw.get("max_sequence_length", 1024)),
            device=str(raw.get("device", "cpu")),
            seed=int(raw.get("seed", 42)),
            llama_cpp_path=_path(raw.get("llama_cpp_path"), "llama_cpp_path", True),
        )
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        raise QuantizationConfigError(f"invalid configuration: {exc}") from exc
    config.validate()
    return config
