"""Real GPU Phase 3B Stage 1 runner built on the Phase 3A abstractions."""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from clauseforge.evaluation.metrics import classification_metrics
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.checkpoints import resume_adapter, save_adapter, write_json
from clauseforge.training.compatibility import checkpoint_evaluation_compatibility
from clauseforge.training.config import TrainingConfig
from clauseforge.training.dataset import TrainingDataset, build_training_dataset
from clauseforge.training.diagnostics import (
    ValidationPrediction,
    aggregate_diagnostics,
    build_prediction,
    write_predictions,
)
from clauseforge.training.environment import environment_metadata
from clauseforge.training.lora import attach_lora, prepare_qlora_base
from clauseforge.training.model import load_production_model
from clauseforge.training.pilot import (
    PILOT_LABEL,
    build_pilot_dataset,
    combined_selection_checksum,
    diagnostic_validation_selection,
    restore_pilot_selection,
)
from clauseforge.training.targets import (
    TargetRepresentation,
    UnknownTargetError,
    resolve_generated_target,
    stable_id_map_checksum,
)
from clauseforge.training.templates import TrainingExample, render_prompt
from clauseforge.training.trainer import build_training_batch, seed_everything

SANITY_LABEL = "GPU SANITY RUN — NOT MODEL PERFORMANCE"
QWEN25_7B_ATTENTION_LORA_PARAMETERS_PER_RANK = 630_784


@dataclass(frozen=True, slots=True)
class ResumeState:
    global_step: int
    examples_seen: int
    epoch: int
    experiment_id: str
    run_mode: str = "full"
    subset_checksum: str = ""


def expected_qwen_attention_trainable_parameters(rank: int) -> int:
    """Return LoRA A/B parameters for 28 Qwen attention blocks at one rank."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    return QWEN25_7B_ATTENTION_LORA_PARAMETERS_PER_RANK * rank


def interrupted_run_metadata(
    exc: BaseException, state: ResumeState
) -> dict[str, object]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "global_step": state.global_step,
        "examples_seen": state.examples_seen,
        "resume_supported": True,
    }


def _numeric_metric(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def configure_gradient_checkpointing(model: Any, config: TrainingConfig) -> None:
    """Enable checkpointing with the explicit non-reentrant PyTorch behavior."""
    if config.optimization.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": (
                    config.optimization.gradient_checkpointing_use_reentrant
                )
            }
        )
    model.config.use_cache = config.model.use_cache


def _optimizer(model: Any, config: TrainingConfig) -> torch.optim.Optimizer:
    parameters = (item for item in model.parameters() if item.requires_grad)
    if config.optimization.optimizer == "paged_adamw_8bit":
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise RuntimeError("paged_adamw_8bit requires bitsandbytes") from exc
        return bnb.optim.PagedAdamW8bit(
            parameters,
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, config: TrainingConfig
) -> Any:
    warmup = int(total_steps * config.optimization.warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        if config.optimization.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _checkpoint(
    experiment_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: ResumeState,
) -> Path:
    checkpoint = experiment_dir / f"checkpoint-{state.global_step}"
    save_adapter(model, checkpoint)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "resume_state": asdict(state),
        },
        checkpoint / "training_state.pt",
    )
    write_json(checkpoint / "resume_state.json", asdict(state))
    return checkpoint


def _append_log(path: Path, event: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def _load_resume(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_experiment_id: str,
    expected_run_mode: str,
    expected_subset_checksum: str,
) -> ResumeState:
    payload = torch.load(
        path / "training_state.pt", map_location="cpu", weights_only=True
    )
    raw = payload["resume_state"]
    state = ResumeState(**raw)
    validate_resume_compatibility(
        state,
        expected_experiment_id,
        expected_run_mode,
        expected_subset_checksum,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return state


def validate_resume_compatibility(
    state: ResumeState,
    expected_experiment_id: str,
    expected_run_mode: str,
    expected_subset_checksum: str,
) -> None:
    if state.experiment_id != expected_experiment_id:
        raise ValueError("checkpoint belongs to a different experiment configuration")
    if (
        state.run_mode != expected_run_mode
        or state.subset_checksum != expected_subset_checksum
    ):
        raise ValueError("checkpoint pilot mode or selected examples are incompatible")


def _generate_label(
    model: Any,
    tokenizer: Any,
    example: TrainingExample,
    taxonomy: tuple[str, ...],
    max_length: int,
    max_new_tokens: int,
    target_representation: TargetRepresentation = "canonical_question",
) -> tuple[str | None, ValidationPrediction]:
    inputs = tokenizer(
        render_prompt(example.clause_text, example.prompt_template_version),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device_inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **device_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0, device_inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    prediction_record = build_prediction(
        clause_id=example.clause_id,
        canonical_target=example.target_label,
        raw_generated_text=raw,
        generated_token_count=int(new_tokens.numel()),
        target_token_count=len(
            tokenizer.encode(example.target_label, add_special_tokens=False)
        ),
        generation_limit=max_new_tokens,
        target_representation=target_representation,
    )
    try:
        accepted: str | None = resolve_generated_target(
            raw, target_representation
        ).canonical
        if accepted not in taxonomy:
            accepted = None
    except UnknownTargetError:
        accepted = None
    return accepted, prediction_record


def validation_metrics(
    model: Any,
    tokenizer: Any,
    dataset: TrainingDataset,
    max_length: int,
    max_new_tokens: int = 160,
    predictions_path: Path | None = None,
    target_representation: TargetRepresentation = "canonical_question",
) -> dict[str, object]:
    predictions: list[str] = []
    expected: list[str] = []
    statuses: Counter[str] = Counter()
    records: list[ValidationPrediction] = []
    for example in dataset.validation:
        prediction, record = _generate_label(
            model,
            tokenizer,
            example,
            dataset.taxonomy,
            max_length,
            max_new_tokens,
            target_representation,
        )
        statuses[record.validation_status] += 1
        records.append(record)
        predictions.append(prediction or "__INVALID_GENERATION__")
        expected.append(record.canonical_target)
    metrics = classification_metrics(expected, predictions, list(dataset.taxonomy))
    invalid_count = statuses["invalid"] + statuses["empty"] + statuses["malformed"]
    if predictions_path is not None:
        write_predictions(predictions_path, records)
    return {
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "total": len(dataset.validation),
        "exact_outputs": statuses["exact"],
        "exact_id_outputs": sum(
            record.status_reason == "exact_id_match" for record in records
        ),
        "invalid_outputs": statuses["invalid"],
        "empty_outputs": statuses["empty"],
        "malformed_outputs": statuses["malformed"],
        "invalid_output_rate": invalid_count / len(dataset.validation),
        "max_new_tokens": max_new_tokens,
        "output_diagnostics": aggregate_diagnostics(records),
        "prediction_distribution": dict(Counter(predictions)),
        "target_representation": target_representation,
        "stable_id_map_checksum": stable_id_map_checksum(),
        "test_evaluated": False,
    }


def validation_loss(
    model: Any, tokenizer: Any, dataset: TrainingDataset, max_length: int
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in dataset.validation:
            batch = build_training_batch(example, tokenizer, max_length)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach()))
    return sum(losses) / len(losses)


def evaluate_phase3b_checkpoint(
    config: TrainingConfig,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    pilot: bool = False,
    pilot_validation_examples: int | None = None,
) -> dict[str, object]:
    """Evaluate an explicitly provisioned checkpoint on validation only."""
    source_dataset = build_training_dataset(
        data_dir,
        max_train_examples=config.data.max_train_examples,
        max_validation_examples=config.data.max_validation_examples,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    experiment_dir = checkpoint.parent

    def load_metadata(name: str, root: Path = experiment_dir) -> dict[str, object]:
        metadata_path = root / name
        if not metadata_path.is_file():
            raise ValueError(f"checkpoint lineage metadata is missing: {name}")
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint lineage metadata is malformed: {name}")
        return cast(dict[str, object], value)

    historical_config = load_metadata("experiment_config.json")
    pilot_config = load_metadata("pilot_config.json") if pilot else {}
    resume_metadata = load_metadata("resume_state.json", checkpoint)
    adapter_metadata = load_metadata("adapter_metadata.json")
    historical_dataset = load_metadata("dataset_manifest.json")
    historical_taxonomy = historical_dataset.get("taxonomy")
    if not isinstance(historical_taxonomy, list) or not all(
        isinstance(item, str) for item in historical_taxonomy
    ):
        raise ValueError("checkpoint taxonomy lineage is malformed")
    dataset = source_dataset
    subset_checksum = ""
    historical_train_summary: dict[str, object] = {}
    historical_validation_summary: dict[str, object] = {}
    evaluation_validation_summary: dict[str, object] = {}
    evaluation_override = False
    if pilot:
        train_count = pilot_config.get("train_examples")
        validation_count = pilot_config.get("validation_examples")
        seed = pilot_config.get("seed")
        if not all(
            isinstance(item, int) for item in (train_count, validation_count, seed)
        ):
            raise ValueError("persisted pilot sample configuration is malformed")
        selected_train_path = experiment_dir / "selected_train_examples.json"
        selected_validation_path = experiment_dir / "selected_validation_examples.json"
        if selected_train_path.is_file() and selected_validation_path.is_file():
            train_selection = restore_pilot_selection(
                source_dataset.train,
                load_metadata("selected_train_examples.json"),
                split="train",
            )
            validation_selection = restore_pilot_selection(
                source_dataset.validation,
                load_metadata("selected_validation_examples.json"),
                split="validation",
            )
        elif selected_train_path.exists() or selected_validation_path.exists():
            raise ValueError("persisted pilot selection metadata is incomplete")
        else:
            _, train_selection, validation_selection = build_pilot_dataset(
                source_dataset,
                train_count=cast(int, train_count),
                validation_count=cast(int, validation_count),
                seed=cast(int, seed),
            )
        subset_checksum = combined_selection_checksum(
            train_selection, validation_selection
        )
        historical_train_summary = train_selection.metadata()
        historical_validation_summary = validation_selection.metadata()
        evaluation_selection, evaluation_override = diagnostic_validation_selection(
            source_dataset.validation,
            validation_selection,
            requested_count=pilot_validation_examples,
            seed=config.seed,
        )
        evaluation_validation_summary = evaluation_selection.metadata()
        dataset = TrainingDataset(
            train_selection.selected,
            evaluation_selection.selected,
            source_dataset.taxonomy,
            source_dataset.manifest,
        )
    compatibility = checkpoint_evaluation_compatibility(
        config,
        historical_config,
        pilot_config,
        resume_metadata,
        adapter_metadata,
        expected_run_mode="pilot" if pilot else "full",
        expected_subset_checksum=subset_checksum,
        historical_taxonomy=historical_taxonomy,
        current_taxonomy=dataset.taxonomy,
        historical_training_subset=historical_train_summary,
        historical_validation_subset=historical_validation_summary,
        evaluation_validation_subset=evaluation_validation_summary,
        evaluation_override=evaluation_override,
    )
    if not compatibility.compatible:
        fields = ", ".join(compatibility.blocking_differences)
        if "selected_subset_checksum" in compatibility.blocking_differences:
            fields += (
                f" (stored={resume_metadata.get('subset_checksum')}, "
                f"calculated={subset_checksum})"
            )
        raise ValueError(f"checkpoint evaluation is incompatible: {fields}")
    model, tokenizer = load_production_model(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = resume_adapter(prepare_qlora_base(model), checkpoint)
    configure_gradient_checkpointing(model, config)
    model.eval()
    predictions_path = output_dir / "validation_predictions.jsonl"
    metrics = validation_metrics(
        model,
        tokenizer,
        dataset,
        config.data.max_sequence_length,
        config.data.validation_max_new_tokens,
        predictions_path,
        config.target_representation,
    )
    metrics.update(
        {
            "checkpoint_id": checkpoint.name,
            "pilot": pilot,
            "prediction_artifact": predictions_path.name,
            "prompt_template_version": config.prompt_template_version,
            "checkpoint_training_lineage": {
                "experiment_id": resume_metadata["experiment_id"],
                "run_mode": resume_metadata["run_mode"],
                "subset_checksum": resume_metadata["subset_checksum"],
                "model": historical_config["model"],
                "lora": historical_config["lora"],
                "prompt_template_version": historical_config["prompt_template_version"],
                "max_sequence_length": cast(
                    dict[str, object], historical_config["data"]
                )["max_sequence_length"],
            },
            "evaluation_configuration": {
                "validation_max_new_tokens": config.data.validation_max_new_tokens,
                "diagnostics": True,
                "validation_subset_mode": (
                    "new_diagnostic_subset"
                    if evaluation_override
                    else "historical_pilot_subset"
                ),
            },
            "compatibility_report": compatibility.to_dict(),
            "test_evaluated": False,
        }
    )
    write_json(output_dir / "validation_diagnostics.json", metrics)
    return metrics


def run_phase3b(
    config: TrainingConfig,
    data_dir: Path,
    *,
    sanity_steps: int | None = None,
    resume_from_checkpoint: Path | None = None,
    pilot: bool = False,
    pilot_train_examples: int = 512,
    pilot_validation_examples: int = 256,
    max_steps: int | None = None,
) -> dict[str, object]:
    """Run real QLoRA on GPU; callers must invoke this only on the provisioned host."""
    if config.model.family != "qwen2" or config.model.quantization != "4bit":
        raise ValueError("Phase 3B Stage 1 requires the configured Qwen 4-bit model")
    if sanity_steps is not None and sanity_steps <= 0:
        raise ValueError("sanity steps must be positive")
    if pilot and sanity_steps is not None:
        raise ValueError("pilot and sanity modes are mutually exclusive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    dataset = build_training_dataset(
        data_dir,
        max_train_examples=config.data.max_train_examples,
        max_validation_examples=config.data.max_validation_examples,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    subset_checksum = ""
    train_selection = validation_selection = None
    if pilot:
        dataset, train_selection, validation_selection = build_pilot_dataset(
            dataset,
            train_count=pilot_train_examples,
            validation_count=pilot_validation_examples,
            seed=config.seed,
        )
        subset_checksum = combined_selection_checksum(
            train_selection, validation_selection
        )
    experiment_dir = (
        config.output_dir / "pilot" / config.experiment_id
        if pilot
        else config.output_dir / config.experiment_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "experiment_config.json", config.to_dict())
    write_json(experiment_dir / "dataset_manifest.json", dataset.manifest)
    write_json(experiment_dir / "environment.json", environment_metadata())
    if pilot and train_selection is not None and validation_selection is not None:
        pilot_config = {
            "label": PILOT_LABEL,
            "train_examples": pilot_train_examples,
            "validation_examples": pilot_validation_examples,
            "seed": config.seed,
            "max_steps": max_steps,
            "checkpoint_steps": 5,
            "validation_max_new_tokens": config.data.validation_max_new_tokens,
            "selection_checksum": subset_checksum,
            "target_representation": config.target_representation,
            "target_representation_version": config.target_representation_version,
            "prompt_template_version": config.prompt_template_version,
            "taxonomy_version": TAXONOMY_VERSION,
            "stable_id_map_checksum": stable_id_map_checksum(),
            "test_evaluated": False,
        }
        write_json(experiment_dir / "pilot_config.json", pilot_config)
        write_json(
            experiment_dir / "selected_train_examples.json",
            train_selection.metadata(),
        )
        write_json(
            experiment_dir / "selected_validation_examples.json",
            validation_selection.metadata(),
        )
    seed_everything(config.seed)
    started = time.perf_counter()
    model, tokenizer = load_production_model(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        resume_adapter(prepare_qlora_base(model), resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else attach_lora(model, config.model, config.lora)
    )
    configure_gradient_checkpointing(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = _optimizer(model, config)
    accumulation = config.optimization.gradient_accumulation
    total_steps = (
        math.ceil(len(dataset.train) / accumulation) * config.optimization.epochs
    )
    scheduler = _scheduler(optimizer, total_steps, config)
    run_mode = "pilot" if pilot else "full"
    state = ResumeState(0, 0, 0, config.experiment_id, run_mode, subset_checksum)
    if resume_from_checkpoint is not None:
        state = _load_resume(
            resume_from_checkpoint,
            optimizer,
            scheduler,
            config.experiment_id,
            run_mode,
            subset_checksum,
        )
    losses: list[float] = []
    log_path = experiment_dir / "training_log.jsonl"
    best_validation_macro_f1 = -1.0
    best_metadata_path = experiment_dir / "best_validation" / "selection_metadata.json"
    if resume_from_checkpoint is not None and best_metadata_path.is_file():
        previous_best = json.loads(best_metadata_path.read_text(encoding="utf-8"))
        best_validation_macro_f1 = float(previous_best["macro_f1"])
    model.train()
    optimizer.zero_grad()
    try:
        stop = False
        for epoch in range(state.epoch, config.optimization.epochs):
            for index, example in enumerate(dataset.train):
                absolute = epoch * len(dataset.train) + index
                if absolute < state.examples_seen:
                    continue
                batch = build_training_batch(
                    example, tokenizer, config.data.max_sequence_length
                )
                batch = {key: value.to(model.device) for key, value in batch.items()}
                output = model(**batch)
                (output.loss / accumulation).backward()
                losses.append(float(output.loss.detach()))
                examples_seen = absolute + 1
                if examples_seen % accumulation == 0 or index + 1 == len(dataset.train):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    state = ResumeState(
                        state.global_step + 1,
                        examples_seen,
                        epoch,
                        config.experiment_id,
                        run_mode,
                        subset_checksum,
                    )
                    checkpoint_steps = 5 if pilot else config.optimization.save_steps
                    if state.global_step % checkpoint_steps == 0:
                        _checkpoint(experiment_dir, model, optimizer, scheduler, state)
                    _append_log(
                        log_path,
                        {
                            "event": "optimizer_step",
                            "global_step": state.global_step,
                            "examples_seen": state.examples_seen,
                            "loss": losses[-1],
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                    )
                    if (
                        sanity_steps is None
                        and state.global_step % config.optimization.eval_steps == 0
                    ):
                        measured = validation_metrics(
                            model,
                            tokenizer,
                            dataset,
                            config.data.max_sequence_length,
                            config.data.validation_max_new_tokens,
                            target_representation=config.target_representation,
                        )
                        measured["global_step"] = state.global_step
                        _append_log(log_path, {"event": "validation", **measured})
                        score = _numeric_metric(measured["macro_f1"], "macro_f1")
                        if score > best_validation_macro_f1:
                            best_validation_macro_f1 = score
                            save_adapter(model, experiment_dir / "best_validation")
                            write_json(
                                experiment_dir
                                / "best_validation"
                                / "selection_metadata.json",
                                {
                                    "split": "validation",
                                    "macro_f1": score,
                                    "global_step": state.global_step,
                                    "test_evaluated": False,
                                },
                            )
                        model.train()
                    if sanity_steps is not None and state.global_step >= sanity_steps:
                        stop = True
                        break
                    if max_steps is not None and state.global_step >= max_steps:
                        stop = True
                        break
            if stop:
                break
            state = ResumeState(
                state.global_step,
                state.examples_seen,
                epoch + 1,
                config.experiment_id,
                run_mode,
                subset_checksum,
            )
        checkpoint = _checkpoint(experiment_dir, model, optimizer, scheduler, state)
        runtime = time.perf_counter() - started
        adapter_dir = save_adapter(model, experiment_dir)
        training = {
            "label": (
                SANITY_LABEL
                if sanity_steps is not None
                else PILOT_LABEL
                if pilot
                else "PHASE 3B STAGE 1 TRAINING"
            ),
            "training_steps": state.global_step,
            "examples_seen": state.examples_seen,
            "training_loss": sum(losses) / len(losses),
            "runtime_seconds": runtime,
            "examples_per_second": state.examples_seen / runtime,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(0),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "adapter_size_bytes": sum(
                path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
            ),
            "checkpoint": str(checkpoint),
            "test_evaluated": False,
            "pilot": pilot,
            "category_coverage": (
                train_selection.metadata() if train_selection is not None else None
            ),
        }
        write_json(
            experiment_dir / "adapter_metadata.json",
            {
                "adapter_only": True,
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "target_modules": list(config.lora.target_modules),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "adapter_size_bytes": training["adapter_size_bytes"],
                "target_representation": config.target_representation,
                "target_representation_version": config.target_representation_version,
                "prompt_template_version": config.prompt_template_version,
                "stable_id_map_checksum": stable_id_map_checksum(),
                "taxonomy_version": TAXONOMY_VERSION,
            },
        )
        write_json(experiment_dir / "training_metrics.json", training)
        validation = (
            {"skipped": True, "reason": SANITY_LABEL, "test_evaluated": False}
            if sanity_steps is not None
            else validation_metrics(
                model,
                tokenizer,
                dataset,
                config.data.max_sequence_length,
                config.data.validation_max_new_tokens,
                experiment_dir / "validation_predictions.jsonl",
                config.target_representation,
            )
        )
        if sanity_steps is None:
            validation["validation_loss"] = validation_loss(
                model, tokenizer, dataset, config.data.max_sequence_length
            )
            validation["label"] = PILOT_LABEL if pilot else "PHASE 3B STAGE 1"
            validation["category_coverage"] = (
                validation_selection.metadata()
                if validation_selection is not None
                else None
            )
        if sanity_steps is None:
            score = _numeric_metric(validation["macro_f1"], "macro_f1")
            if score > best_validation_macro_f1:
                save_adapter(model, experiment_dir / "best_validation")
                write_json(
                    experiment_dir / "best_validation" / "selection_metadata.json",
                    {
                        "split": "validation",
                        "macro_f1": score,
                        "global_step": state.global_step,
                        "test_evaluated": False,
                    },
                )
        write_json(experiment_dir / "environment.json", environment_metadata())
        write_json(experiment_dir / "validation_metrics.json", validation)
        return {
            "training": training,
            "validation": validation,
            "experiment_dir": str(experiment_dir),
        }
    except BaseException as exc:
        failure = interrupted_run_metadata(exc, state)
        write_json(experiment_dir / "failure.json", failure)
        if pilot:
            write_json(experiment_dir / "failure_metadata.json", failure)
        raise


def select_rank_winner(results: list[dict[str, object]]) -> dict[str, object]:
    """Select highest validation macro F1, breaking exact ties toward lower rank."""
    if not results:
        raise ValueError("rank selection requires completed validation results")
    for result in results:
        score = result.get("macro_f1")
        if (
            result.get("split") != "validation"
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("rank results require real validation macro_f1 values")

    def selection_key(row: dict[str, object]) -> tuple[float, int]:
        score, rank = row["macro_f1"], row.get("rank")
        if not isinstance(score, int | float) or not isinstance(rank, int):
            raise ValueError("rank results require numeric rank and macro_f1")
        return float(score), -rank

    return max(results, key=selection_key)
