"""Training command with validation-first dry run and tiny local smoke mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clauseforge.models.transformer_classifier import (
    UnknownGeneratedLabelError,
    normalize_generated_label,
)
from clauseforge.training.checkpoints import reload_adapter, save_adapter, write_json
from clauseforge.training.config import TrainingConfig, load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.environment import environment_metadata
from clauseforge.training.lora import attach_lora
from clauseforge.training.model import tiny_smoke_model
from clauseforge.training.overfit import run_overfit_diagnostic
from clauseforge.training.phase3b import run_phase3b
from clauseforge.training.pilot import build_pilot_dataset
from clauseforge.training.smoke import build_smoke_tokenizer
from clauseforge.training.templates import render_example
from clauseforge.training.tokenization import analyze_token_lengths
from clauseforge.training.trainer import seed_everything, train_adapter


def _summary(config: TrainingConfig, data_dir: Path) -> dict[str, object]:
    return {
        "experiment_id": config.experiment_id,
        "data_dir": str(data_dir.resolve()),
        "model": config.model.name,
        "revision": config.model.revision,
        "license": config.model.license,
        "quantization": config.model.quantization,
        "max_sequence_length": config.data.max_sequence_length,
        "validation_max_new_tokens": config.data.validation_max_new_tokens,
        "target_representation": config.target_representation,
        "target_representation_version": config.target_representation_version,
        "prompt_template_version": config.prompt_template_version,
        "output_dir": str(config.output_dir.resolve()),
    }


def run(
    config: TrainingConfig,
    data_dir: Path,
    *,
    dry_run: bool,
    sanity_steps: int | None = None,
    resume_from_checkpoint: Path | None = None,
    pilot: bool = False,
    pilot_train_examples: int | None = None,
    pilot_validation_examples: int | None = None,
    max_steps: int | None = None,
    overfit_diagnostic: bool = False,
    overfit_examples: int = 8,
    overfit_steps: int = 100,
    overfit_eval_every: int = 10,
) -> dict[str, object]:
    config.validate()
    if overfit_diagnostic and (pilot or sanity_steps is not None):
        raise ValueError("overfit, pilot, and sanity modes are mutually exclusive")
    if pilot_train_examples is None:
        pilot_train_examples = (
            256 if config.target_representation == "category_id" else 512
        )
    if pilot_validation_examples is None:
        pilot_validation_examples = (
            128 if config.target_representation == "category_id" else 256
        )
    dataset = build_training_dataset(
        data_dir,
        max_train_examples=config.data.max_train_examples,
        max_validation_examples=config.data.max_validation_examples,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    if config.prompt_template_version != dataset.manifest["template_version"]:
        raise ValueError("configured and implemented prompt template versions differ")
    corpus = [render_example(item) for item in dataset.train + dataset.validation]
    if config.model.name != "local/tiny-gpt2":
        if dry_run:
            pilot_manifest: dict[str, object] | None = None
            if pilot:
                pilot_dataset, _, _ = build_pilot_dataset(
                    dataset,
                    train_count=pilot_train_examples,
                    validation_count=pilot_validation_examples,
                    seed=config.seed,
                )
                pilot_manifest = pilot_dataset.manifest
            return {
                "status": (
                    "phase3b-overfit-config-valid"
                    if overfit_diagnostic
                    else "phase3b-pilot-config-valid"
                    if pilot
                    else "phase3b-config-valid"
                ),
                "summary": _summary(config, data_dir),
                "dataset_manifest": pilot_manifest or dataset.manifest,
                "network_or_model_access": False,
                "test_evaluated": False,
            }
        if overfit_diagnostic:
            return run_overfit_diagnostic(
                config,
                data_dir,
                examples_count=overfit_examples,
                steps=overfit_steps,
                eval_every=overfit_eval_every,
            )
        return run_phase3b(
            config,
            data_dir,
            sanity_steps=sanity_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            pilot=pilot,
            pilot_train_examples=pilot_train_examples,
            pilot_validation_examples=pilot_validation_examples,
            max_steps=max_steps,
        )
    tokenizer = build_smoke_tokenizer(corpus)
    report = analyze_token_lengths(
        dataset.train, tokenizer, config.data.max_sequence_length
    )
    result: dict[str, object] = {
        "status": "dry-run-valid" if dry_run else "smoke-complete",
        "summary": _summary(config, data_dir),
        "dataset_manifest": dataset.manifest,
        "token_length_report": report.to_dict(),
    }
    if dry_run:
        return result

    experiment_dir = config.output_dir / config.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(experiment_dir / "tokenizer")
    seed_everything(config.seed)
    base = tiny_smoke_model(len(tokenizer))
    model = attach_lora(base, config.model, config.lora)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    metrics = train_adapter(
        model,
        tokenizer,
        dataset.train,
        dataset.validation,
        config.optimization,
        config.data.max_sequence_length,
        config.seed,
    )
    adapter_dir = save_adapter(model, experiment_dir)
    reloaded = reload_adapter(tiny_smoke_model(len(tokenizer)), adapter_dir)
    inference_inputs = tokenizer(
        render_example(dataset.validation[0]),
        return_tensors="pt",
        truncation=True,
        max_length=config.data.max_sequence_length - 4,
    )
    generated = reloaded.generate(
        **inference_inputs,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[0, inference_inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    accepted_label: str | None = None
    validation_error: str | None = None
    try:
        accepted_label = normalize_generated_label(generated_text, dataset.taxonomy)
    except UnknownGeneratedLabelError as exc:
        validation_error = str(exc)
    write_json(experiment_dir / "experiment_config.json", config.to_dict())
    write_json(experiment_dir / "training_metrics.json", metrics.to_dict())
    write_json(experiment_dir / "dataset_manifest.json", dataset.manifest)
    write_json(experiment_dir / "environment.json", environment_metadata())
    result["metrics"] = metrics.to_dict()
    result["checkpoint_reload_verified"] = True
    result["lora_parameters"] = {
        "trainable": trainable,
        "total": total,
        "trainable_percentage": 100.0 * trainable / total,
    }
    result["inference"] = {
        "adapter_generated_text": generated_text,
        "accepted_exact_label": accepted_label,
        "validation_error": validation_error,
        "behavior": "unknown output rejected; smoke quality is not evaluated",
    }
    result["experiment_dir"] = str(experiment_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sanity-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-train-examples", type=int)
    parser.add_argument("--pilot-validation-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overfit-diagnostic", action="store_true")
    parser.add_argument("--overfit-examples", type=int, default=8)
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument("--overfit-eval-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(json.dumps(_summary(config, args.data), indent=2, sort_keys=True))
    print(
        json.dumps(
            run(
                config,
                args.data,
                dry_run=args.dry_run,
                sanity_steps=args.sanity_steps,
                resume_from_checkpoint=args.resume_from_checkpoint,
                pilot=args.pilot,
                pilot_train_examples=args.pilot_train_examples,
                pilot_validation_examples=args.pilot_validation_examples,
                max_steps=args.max_steps,
                overfit_diagnostic=args.overfit_diagnostic,
                overfit_examples=args.overfit_examples,
                overfit_steps=args.overfit_steps,
                overfit_eval_every=args.overfit_eval_every,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0
