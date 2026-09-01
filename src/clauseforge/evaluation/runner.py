"""Baseline training, evaluation, and artifact orchestration."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import sklearn

from clauseforge.baselines.base import ClassifierProtocol, FloatMatrix
from clauseforge.baselines.majority import MajorityClassifier
from clauseforge.baselines.rules import KeywordRuleClassifier
from clauseforge.baselines.tfidf import TfidfLinearClassifier
from clauseforge.data.manifest import sha256_file
from clauseforge.evaluation.bootstrap import bootstrap_confidence_intervals
from clauseforge.evaluation.calibration import calibration_metrics
from clauseforge.evaluation.confusion import confusion_analysis
from clauseforge.evaluation.dataset import (
    EvaluationDataError,
    EvaluationDataset,
    EvaluationExample,
    load_evaluation_dataset,
)
from clauseforge.evaluation.errors import build_errors, summarize_errors
from clauseforge.evaluation.metrics import classification_metrics, top_k_recall

MODEL_NAMES = ("majority", "rules", "tfidf-logreg", "tfidf-svm")
LOCKED_C = {"tfidf-logreg": 2.0, "tfidf-svm": 0.5}


def build_model(name: str, *, c_override: float | None = None) -> ClassifierProtocol:
    if name == "majority":
        return MajorityClassifier()
    if name == "rules":
        return KeywordRuleClassifier()
    if name == "tfidf-logreg":
        return TfidfLinearClassifier("logreg", c=c_override or LOCKED_C[name])
    if name == "tfidf-svm":
        return TfidfLinearClassifier("svm", c=c_override or LOCKED_C[name])
    raise ValueError(f"unknown model: {name}")


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


def _confidence_values(
    model: ClassifierProtocol, texts: list[str], scores: FloatMatrix
) -> tuple[list[float | None], FloatMatrix | None]:
    probabilities = model.predict_proba(texts)
    if probabilities is not None:
        return [float(value) for value in probabilities.max(axis=1)], probabilities
    return [float(value) for value in scores.max(axis=1)], None


def _class_distribution(
    examples: list[EvaluationExample], labels: list[str]
) -> dict[str, object]:
    counts = Counter(example.label for example in examples)
    supports = sorted(counts.get(label, 0) for label in labels)
    nonzero = [value for value in supports if value > 0]
    return {
        "counts": {label: counts.get(label, 0) for label in labels},
        "minimum_support": min(supports),
        "median_support": float(np.median(supports)),
        "maximum_support": max(supports),
        "imbalance_ratio_nonzero": max(nonzero) / min(nonzero),
        "zero_support_categories": [
            label for label in labels if counts.get(label, 0) == 0
        ],
        "categories_below_5": [label for label in labels if counts.get(label, 0) < 5],
        "categories_below_10": [label for label in labels if counts.get(label, 0) < 10],
    }


def evaluate_model(
    model: ClassifierProtocol,
    dataset: EvaluationDataset,
    split: str,
    output_dir: Path,
    *,
    bootstrap_iterations: int = 500,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    train = dataset.split("train")
    evaluation = dataset.split(split)
    model.fit([item.text for item in train], [item.label for item in train])
    texts = [item.text for item in evaluation]
    y_true = [item.label for item in evaluation]
    predictions_array = model.predict(texts)
    predictions = predictions_array.tolist()
    scores = model.predict_scores(texts)
    classes = model.classes_.tolist()
    metrics = classification_metrics(y_true, predictions, dataset.labels)
    metrics["top_k_recall"] = {
        f"recall_at_{k}": top_k_recall(y_true, scores, classes, k) for k in (1, 3, 5)
    }
    confidence, probabilities = _confidence_values(model, texts, scores)
    bootstrap = bootstrap_confidence_intervals(
        y_true,
        predictions,
        [item.contract_id for item in evaluation],
        dataset.labels,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    confusion = confusion_analysis(y_true, predictions, dataset.labels)
    errors = build_errors(
        evaluation, predictions, confidence, split=split, model_name=model.name
    )
    calibration = (
        calibration_metrics(y_true, probabilities, classes)
        if probabilities is not None
        else {"available": False, "reason": "model does not expose probabilities"}
    )
    high_confidence_errors = (
        sum(
            error["confidence"] is not None and cast(float, error["confidence"]) >= 0.8
            for error in errors
        )
        if probabilities is not None
        else 0
    )
    per_class = metrics.pop("per_class")
    summary = {
        "model": model.name,
        "split": split,
        "example_count": len(evaluation),
        "contract_count": len({item.contract_id for item in evaluation}),
        "metrics": metrics,
        "error_count": len(errors),
        "high_confidence_error_count_probability_models_only": high_confidence_errors,
        "configuration": model.configuration(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "summary.json", summary)
    _json(output_dir / "metrics.json", metrics)
    _json(output_dir / "per_class_metrics.json", per_class)
    _json(output_dir / "confusion_matrix.json", confusion)
    _jsonl(output_dir / "error_analysis.jsonl", errors)
    _json(
        output_dir / "error_summary.json",
        summarize_errors(errors, probability_confidence=probabilities is not None),
    )
    _json(output_dir / "calibration.json", calibration)
    _json(output_dir / "bootstrap.json", bootstrap)
    return summary


def run(
    data_dir: Path,
    output_dir: Path,
    *,
    models: list[str],
    split: str,
    c_override: float | None = None,
    bootstrap_iterations: int = 500,
    reuse_existing: bool = False,
) -> list[dict[str, object]]:
    dataset = load_evaluation_dataset(data_dir)
    summaries: list[dict[str, object]] = []
    for name in models:
        summary_path = output_dir / name / split / "summary.json"
        if reuse_existing:
            if not summary_path.is_file():
                raise ValueError(f"existing result is missing: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError(f"existing summary is malformed: {summary_path}")
            errors_path = summary_path.parent / "error_analysis.jsonl"
            errors = [
                json.loads(line)
                for line in errors_path.read_text(encoding="utf-8").splitlines()
            ]
            _json(
                summary_path.parent / "error_summary.json",
                summarize_errors(
                    errors, probability_confidence=name in {"majority", "tfidf-logreg"}
                ),
            )
            summaries.append(summary)
            continue
        model = build_model(name, c_override=c_override)
        summaries.append(
            evaluate_model(
                model,
                dataset,
                split,
                output_dir / name / split,
                bootstrap_iterations=bootstrap_iterations,
            )
        )
        del model
        gc.collect()
    run_config = {
        "data_dir": str(data_dir.resolve()),
        "dataset_version": "aok_v1.0",
        "processing_version": "1.0.0",
        "split": split,
        "models": models,
        "locked_c": LOCKED_C,
        "bootstrap": {
            "unit": "contract",
            "iterations": bootstrap_iterations,
            "seed": 42,
        },
        "checksums": {
            name: sha256_file(data_dir / name)
            for name in (
                "contracts.jsonl",
                "clauses.jsonl",
                "splits.json",
                "manifest.json",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_label_tuning": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / f"run_config_{split}.json", run_config)
    _json(output_dir / f"comparison_{split}.json", summaries)
    _write_results_markdown(
        output_dir / f"RESULTS_{split.upper()}.md", summaries, split
    )
    return summaries


def _write_results_markdown(
    path: Path, summaries: list[dict[str, object]], split: str
) -> None:
    lines = [
        f"# Classical baseline results: {split}",
        "",
        "Generated from actual local execution. "
        "CUAD categories are authoritative labels.",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Recall@3 | Recall@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary["metrics"]
        assert isinstance(metrics, dict)
        top_k = metrics["top_k_recall"]
        assert isinstance(top_k, dict)
        lines.append(
            f"| {summary['model']} | {float(metrics['accuracy']):.6f} | "
            f"{float(metrics['macro_f1']):.6f} | {float(metrics['weighted_f1']):.6f} | "
            f"{float(top_k['recall_at_3']):.6f} | {float(top_k['recall_at_5']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ClauseForge classification baselines"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--c", type=float, default=None, help="validation selection override"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="aggregate already-generated model summaries without refitting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    try:
        run(
            args.data,
            args.output,
            models=models,
            split=args.split,
            c_override=args.c,
            bootstrap_iterations=args.bootstrap_iterations,
            reuse_existing=args.reuse_existing,
        )
    except (EvaluationDataError, OSError, ValueError) as exc:
        print(f"Baseline evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
