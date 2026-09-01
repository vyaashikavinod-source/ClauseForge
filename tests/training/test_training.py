from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clauseforge.training.checkpoints import reload_adapter, save_adapter, write_json
from clauseforge.training.config import ConfigurationError, load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.lora import attach_lora, build_lora_config
from clauseforge.training.model import (
    build_4bit_config,
    require_qlora_runtime,
    tiny_smoke_model,
)
from clauseforge.training.smoke import build_smoke_tokenizer
from clauseforge.training.templates import (
    TrainingExample,
    render_example,
    render_prompt,
)
from clauseforge.training.tokenization import analyze_token_lengths
from clauseforge.training.trainer import train_adapter


def test_config_parsing_and_identity() -> None:
    path = Path("training/configs/smoke.yaml")
    first = load_config(path)
    second = load_config(path)
    assert first.experiment_id == second.experiment_id
    assert first.lora.target_modules == ("c_attn",)


def test_invalid_configuration_rejected() -> None:
    config = load_config(Path("training/configs/smoke.yaml"))
    with pytest.raises(ConfigurationError):
        replace(config, data=replace(config.data, max_sequence_length=8)).validate()


def test_template_is_deterministic_and_preserves_label() -> None:
    example = TrainingExample("c1", "d1", "Notice clause", "Notices", "train")
    assert render_example(example) == render_example(example)
    assert render_example(example).endswith("Assistant:\nNotices")
    assert render_prompt("x").endswith("Assistant:\n")


class LengthTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def test_tokenization_analysis_reports_truncation() -> None:
    examples = (
        TrainingExample("1", "a", "one", "A", "train"),
        TrainingExample("2", "b", " ".join(["word"] * 30), "B", "train"),
    )
    report = analyze_token_lengths(examples, LengthTokenizer(), 25)
    assert report.truncated_count == 1
    assert report.truncated_percentage == 50.0
    assert report.truncation_by_category["B"]["truncated"] == 1


def test_real_dataset_isolation_and_taxonomy() -> None:
    dataset = build_training_dataset(Path("data/processed/cuad/1.0.0-run-a"))
    assert len(dataset.taxonomy) == 41
    assert len(dataset.train) == 11223
    assert len(dataset.validation) == 1324
    assert {x.contract_id for x in dataset.train}.isdisjoint(
        {x.contract_id for x in dataset.validation}
    )
    assert {x.target_label for x in dataset.train} <= set(dataset.taxonomy)


def test_lora_config_and_cpu_qlora_failure() -> None:
    config = load_config(Path("training/configs/smoke.yaml"))
    lora = build_lora_config(config.model, config.lora)
    assert lora.r == 4
    assert "c_attn" in lora.target_modules
    with pytest.raises(RuntimeError, match="CUDA"):
        require_qlora_runtime()


def test_qlora_configuration_is_nf4_with_double_quantization() -> None:
    quantization = build_4bit_config("bfloat16")
    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_use_double_quant is True


def test_checkpoint_metadata_and_reload(tmp_path: Path) -> None:
    config = load_config(Path("training/configs/smoke.yaml"))
    model = attach_lora(tiny_smoke_model(64), config.model, config.lora)
    adapter = save_adapter(model, tmp_path)
    reloaded = reload_adapter(tiny_smoke_model(64), adapter)
    assert reloaded.peft_config
    write_json(tmp_path / "metadata.json", {"verified": True})
    assert json.loads((tmp_path / "metadata.json").read_text())["verified"] is True


def test_tiny_end_to_end_training() -> None:
    config = load_config(Path("training/configs/smoke.yaml"))
    examples = (
        TrainingExample("1", "a", "Notice by mail.", "Notices", "train"),
        TrainingExample("2", "b", "Agreement expires.", "Expiration Date", "train"),
    )
    validation = (TrainingExample("3", "c", "Send notice.", "Notices", "validation"),)
    tokenizer = build_smoke_tokenizer(render_example(x) for x in examples + validation)
    model = attach_lora(tiny_smoke_model(len(tokenizer)), config.model, config.lora)
    metrics = train_adapter(
        model, tokenizer, examples, validation, config.optimization, 128, 42
    )
    assert metrics.steps == 1
    assert metrics.training_loss > 0
    assert metrics.validation_loss > 0
