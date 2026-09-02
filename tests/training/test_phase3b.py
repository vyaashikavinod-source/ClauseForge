from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from clauseforge.taxonomy import (
    category_by_canonical,
    category_by_id,
    load_taxonomy_metadata,
)
from clauseforge.training.cli import build_parser
from clauseforge.training.config import load_config
from clauseforge.training.phase3b import (
    ResumeState,
    checkpoint_lineage_metadata,
    configure_gradient_checkpointing,
    expected_qwen_attention_trainable_parameters,
    interrupted_run_metadata,
    planned_training_schedule,
    scheduled_steps,
    select_best_validation_checkpoint,
    select_rank_winner,
    validate_checkpoint_lineage,
)
from clauseforge.training.trainer import mask_prompt_tokens

CONFIG_DIR = Path("training/configs/phase3b")


def test_taxonomy_metadata_preserves_all_canonical_values() -> None:
    categories = load_taxonomy_metadata()
    assert len(categories) == 41
    assert len({item.category_id for item in categories}) == 41
    renewal = category_by_id("renewal_term")
    assert renewal.category_name == "Renewal Term"
    assert category_by_canonical(renewal.canonical) == renewal
    assert 'related to "Renewal Term"' in renewal.canonical


def test_rank_configs_are_controlled_and_t4_safe() -> None:
    configs = [
        load_config(CONFIG_DIR / f"qwen25_7b_qlora_r{rank}.yaml")
        for rank in (8, 16, 32, 64)
    ]
    reference = configs[0].to_dict()
    for config, rank in zip(configs, (8, 16, 32, 64), strict=True):
        assert config.lora.rank == rank
        assert config.lora.alpha == rank * 2
        assert config.model.precision == "float16"
        assert config.model.quantization == "4bit"
        assert config.model.quant_type == "nf4" and config.model.double_quant
        assert config.model.use_cache is False
        assert config.data.max_sequence_length == 1024
        assert config.optimization.batch_size == 1
        assert config.optimization.gradient_accumulation == 16
        assert config.optimization.gradient_checkpointing
        assert not config.optimization.gradient_checkpointing_use_reentrant
        comparable = config.to_dict()
        comparable["lora"] = reference["lora"]
        assert comparable == reference


def test_trainable_parameter_expectations_match_validated_rank8() -> None:
    assert expected_qwen_attention_trainable_parameters(8) == 5_046_272
    assert [
        expected_qwen_attention_trainable_parameters(rank) for rank in (8, 16, 32, 64)
    ] == [5_046_272, 10_092_544, 20_185_088, 40_370_176]


def test_assistant_only_loss_masking() -> None:
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    labels = mask_prompt_tokens(input_ids, 3)
    assert labels.tolist() == [[-100, -100, -100, 13, 14]]
    assert input_ids.tolist() == [[10, 11, 12, 13, 14]]


def test_gradient_checkpointing_is_explicitly_non_reentrant() -> None:
    class Config:
        use_cache = True

    class Model:
        config = Config()
        kwargs: dict[str, object] | None = None

        def gradient_checkpointing_enable(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    model = Model()
    config = load_config(CONFIG_DIR / "qwen25_7b_qlora_r8.yaml")
    configure_gradient_checkpointing(model, config)
    assert model.kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert model.config.use_cache is False


def test_sanity_and_resume_cli_are_supported() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            str(CONFIG_DIR / "qwen25_7b_qlora_r8.yaml"),
            "--data",
            "data/processed/cuad/1.0.0-run-a",
            "--sanity-steps",
            "1",
            "--resume-from-checkpoint",
            "checkpoints/phase3b/checkpoint-100",
        ]
    )
    assert args.sanity_steps == 1
    assert args.resume_from_checkpoint.name == "checkpoint-100"


def test_resume_and_interruption_metadata_are_inspectable() -> None:
    state = ResumeState(100, 1600, 0, "experiment")
    assert asdict(state)["global_step"] == 100
    failure = interrupted_run_metadata(RuntimeError("disconnect"), state)
    assert failure == {
        "error_type": "RuntimeError",
        "message": "disconnect",
        "global_step": 100,
        "examples_seen": 1600,
        "resume_supported": True,
    }


def test_rank_winner_uses_validation_macro_f1_and_smaller_tie() -> None:
    results = [
        {"rank": 8, "split": "validation", "macro_f1": 0.6},
        {"rank": 16, "split": "validation", "macro_f1": 0.7},
        {"rank": 32, "split": "validation", "macro_f1": 0.7},
    ]
    assert select_rank_winner(results)["rank"] == 16
    with pytest.raises(ValueError, match="real validation"):
        select_rank_winner([{"rank": 8, "split": "test", "macro_f1": 0.9}])
    with pytest.raises(ValueError, match="real validation"):
        select_rank_winner(
            [{"rank": 8, "split": "validation", "macro_f1": float("nan")}]
        )


def _candidate(
    step: int,
    macro_f1: float,
    invalid: float,
    exact: float,
    loss: float,
) -> dict[str, object]:
    return {
        "global_step": step,
        "macro_f1": macro_f1,
        "invalid_output_rate": invalid,
        "exact_id_output_rate": exact,
        "validation_loss": loss,
        "test_evaluated": False,
    }


def test_corrected_v2_pilot_cadence_and_configuration() -> None:
    config = load_config(Path("training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml"))
    assert scheduled_steps(100, config.optimization.save_steps) == (20, 40, 60, 80, 100)
    assert scheduled_steps(100, config.optimization.eval_steps) == (20, 40, 60, 80, 100)
    assert config.target_representation == "category_id"
    assert config.lora.rank == 8 and config.lora.alpha == 16
    assert config.optimization.gradient_accumulation == 16
    assert planned_training_schedule(256, 16, 3, 100) == (100, 7)
    lineage = checkpoint_lineage_metadata(config, "subset-checksum")
    assert lineage["pilot_subset_checksum"] == "subset-checksum"
    assert lineage["target_representation"] == "category_id"
    assert lineage["prompt_template_version"] == "cuad-classification-id-v2"
    assert lineage["stable_id_map_checksum"]
    validate_checkpoint_lineage(lineage, lineage)
    changed = {**lineage, "stable_id_map_checksum": "different"}
    with pytest.raises(ValueError, match="stable_id_map_checksum"):
        validate_checkpoint_lineage(changed, lineage)


def test_best_checkpoint_uses_macro_f1_then_documented_ties() -> None:
    lower_macro = _candidate(20, 0.1, 0.0, 1.0, 0.5)
    best_macro = _candidate(40, 0.2, 0.9, 0.1, 2.0)
    assert select_best_validation_checkpoint([lower_macro, best_macro]) is best_macro

    high_invalid = _candidate(20, 0.2, 0.5, 0.8, 0.5)
    low_invalid = _candidate(40, 0.2, 0.2, 0.1, 2.0)
    assert select_best_validation_checkpoint([high_invalid, low_invalid]) is low_invalid

    low_exact = _candidate(20, 0.2, 0.2, 0.2, 0.5)
    high_exact = _candidate(40, 0.2, 0.2, 0.8, 2.0)
    assert select_best_validation_checkpoint([low_exact, high_exact]) is high_exact

    high_loss = _candidate(20, 0.2, 0.2, 0.8, 2.0)
    low_loss = _candidate(40, 0.2, 0.2, 0.8, 1.0)
    assert select_best_validation_checkpoint([high_loss, low_loss]) is low_loss

    early = _candidate(20, 0.2, 0.2, 0.8, 1.0)
    late = _candidate(40, 0.2, 0.2, 0.8, 1.0)
    assert select_best_validation_checkpoint([late, early]) is early
