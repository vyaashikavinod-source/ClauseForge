"""One frozen benchmark and scorer for frontier, adapter, and quantized servers.

Uses only the standard library and existing CUAD data/target modules. Nothing
downloads a model, spends API credits, or opens the held-out set implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clauseforge.evaluation.dataset import EvaluationExample, load_evaluation_dataset
from clauseforge.training.targets import stable_id_map, target_for_canonical
from clauseforge.training.templates import render_prompt

PROMPT_VERSION = "cuad-classification-id-v3-taxonomy"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def prepare(
    data: Path, output: Path, count: int = 200, seed: int = 42
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("Benchmark count must be positive")
    if output.exists():
        raise ValueError("Frozen benchmark already exists; choose a new path")
    dataset = load_evaluation_dataset(data)
    seen = {
        " ".join(e.text.split()).casefold()
        for split in ("train", "validation")
        for e in dataset.split(split)
    }
    eligible = [
        e
        for e in dataset.split("test")
        if " ".join(e.text.split()).casefold() not in seen
    ]
    # Keep all annotations of a multi-label span together by excluding ambiguous
    # normalized texts from this explicitly single-label benchmark.
    labels_by_text: dict[str, set[str]] = {}
    for e in eligible:
        labels_by_text.setdefault(" ".join(e.text.split()).casefold(), set()).add(
            e.label
        )
    buckets: dict[str, list[EvaluationExample]] = {}
    unique: set[str] = set()
    for e in eligible:
        text_key = " ".join(e.text.split()).casefold()
        if len(labels_by_text[text_key]) == 1 and text_key not in unique:
            buckets.setdefault(e.label, []).append(e)
            unique.add(text_key)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[EvaluationExample] = []
    while len(selected) < count and any(buckets.values()):
        for label in sorted(buckets):
            if buckets[label] and len(selected) < count:
                selected.append(buckets[label].pop())
    if len(selected) != count:
        raise ValueError(f"Need {count} eligible unique spans, found {len(selected)}")
    rows = [
        {**asdict(e), "target": target_for_canonical(e.label, "category_id")}
        for e in selected
    ]
    benchmark = {
        "schema_version": 1,
        "name": "cuad-portfolio-held-out-v1",
        "seed": seed,
        "split": "test",
        "sampling": (
            "category round-robin; contract-disjoint; "
            "exact normalized text decontaminated; ambiguous labels excluded"
        ),
        "prompt_version": PROMPT_VERSION,
        "source_checksums": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(data.glob("*.json*"))
        },
        "labels": [
            {"id": c.category_id, "name": c.category_name} for c in stable_id_map()
        ],
        "examples": rows,
    }
    benchmark["benchmark_sha256"] = digest(benchmark)
    write_json(output, benchmark)
    return {
        "count": len(rows),
        "categories": len({r["target"] for r in rows}),
        "benchmark_sha256": benchmark["benchmark_sha256"],
    }


def score(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty run")
    per_category = {}
    for label in labels:
        tp = sum(r["target"] == label and r["prediction"] == label for r in rows)
        support = sum(r["target"] == label for r in rows)
        predicted = sum(r["prediction"] == label for r in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_category[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "accuracy_within_category": recall if support else None,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    return {
        "sample_size": len(rows),
        "accuracy": sum(r["target"] == r["prediction"] for r in rows) / len(rows),
        "macro_f1_all_categories": statistics.mean(
            v["f1"] for v in per_category.values()
        ),
        "macro_f1_supported_categories": statistics.mean(
            v["f1"] for v in per_category.values() if v["support"]
        ),
        "weighted_f1": sum(v["f1"] * v["support"] for v in per_category.values())
        / len(rows),
        "invalid_output_rate": sum(r["prediction"] not in labels for r in rows)
        / len(rows),
        "request_error_rate": sum(r.get("error") is not None for r in rows) / len(rows),
        "per_category": per_category,
    }


def call_model(
    endpoint: str, model: str, prompt: str, key: str | None, max_tokens: int = 64
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Use HTTPS or a loopback local inference endpoint")
    responses = parsed.path.endswith("/responses")
    payload = (
        {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if responses
        else {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    )
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(), headers=headers
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.load(response)
    if responses:
        text = "".join(
            c.get("text", "")
            for item in body.get("output", [])
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
    else:
        text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "response_model": body.get("model", model),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    interval = getattr(args, "request_interval", 0.0)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("Request interval must be finite and nonnegative")
    if any(
        rate is not None and (not math.isfinite(rate) or rate < 0)
        for rate in (args.input_rate, args.output_rate)
    ):
        raise ValueError("Prices must be finite and nonnegative")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    identity = benchmark.pop("benchmark_sha256")
    if digest(benchmark) != identity:
        raise ValueError("Frozen benchmark checksum mismatch")
    key = os.environ.get(args.key_env)
    if not key and urllib.parse.urlparse(args.endpoint).hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError(
            f"Set {args.key_env} securely before using an external provider"
        )
    if args.max_requests < len(benchmark["examples"]):
        raise ValueError(
            "Request budget is smaller than benchmark; refusing a partial comparison"
        )
    if args.report.exists() or args.report.with_suffix(".jsonl").exists():
        raise ValueError("Use a new report path; never overwrite recorded evidence")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    labels = [c["id"] for c in benchmark["labels"]]
    rows = []
    prior_duration = 0.0
    resume = getattr(args, "resume_from", None)
    if resume:
        prior = json.loads(resume.read_text(encoding="utf-8"))
        journal_bytes = resume.with_suffix(".jsonl").read_bytes()
        if (
            prior["benchmark_sha256"] != identity
            or prior["prompt_version"] != PROMPT_VERSION
            or prior["model"] != args.model
            or prior["model_role"] != args.role
            or prior.get("request_interval_seconds", 0.0) != interval
            or prior["records_sha256"] != hashlib.sha256(journal_bytes).hexdigest()
        ):
            raise ValueError("Resume provenance mismatch")
        rows = [json.loads(line) for line in journal_bytes.decode("utf-8").splitlines()]
        expected = [
            (e["clause_id"], e["target"]) for e in benchmark["examples"][: len(rows)]
        ]
        if (
            [(r["clause_id"], r["target"]) for r in rows] != expected
            or len(rows) >= len(benchmark["examples"])
            or score(rows, labels) != prior["metrics"]
        ):
            raise ValueError(
                "Resume journal is not a verified incomplete benchmark prefix"
            )
        args.report.with_suffix(".jsonl").write_bytes(journal_bytes)
        prior_duration = prior["duration_seconds"]
    start = time.perf_counter()
    previous_start = None
    consecutive_errors = 0
    for example in benchmark["examples"][len(rows) :]:
        if previous_start is not None:
            time.sleep(max(0.0, interval - (time.perf_counter() - previous_start)))
        tick = time.perf_counter()
        previous_start = tick
        row = {
            "clause_id": example["clause_id"],
            "target": example["target"],
            "prediction": None,
            "error": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        try:
            response = call_model(
                args.endpoint,
                args.model,
                render_prompt(example["text"], PROMPT_VERSION),
                key,
            )
            raw = response.pop("text")
            row.update(response)
            row["prediction"] = raw.strip() if raw.strip() in labels else None
            row["raw_output"] = raw[:4000]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Never serialize request/response objects or credentials.
            row["error"] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                row["http_status"] = exc.code
        row["duration_seconds"] = time.perf_counter() - tick
        rows.append(row)
        consecutive_errors = consecutive_errors + 1 if row["error"] else 0
        with args.report.with_suffix(".jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(row) + "\n")
        if row.get("http_status") and getattr(args, "stop_on_http_error", False):
            break
        if getattr(args, "stop_on_access_error", False) and (
            row.get("http_status") in {401, 403, 429} or consecutive_errors >= 3
        ):
            break
    wall = prior_duration + time.perf_counter() - start
    usage_available = all(
        r["input_tokens"] is not None and r["output_tokens"] is not None for r in rows
    )
    input_tokens = sum(r["input_tokens"] for r in rows) if usage_available else None
    output_tokens = sum(r["output_tokens"] for r in rows) if usage_available else None
    cost = (
        (input_tokens * args.input_rate + output_tokens * args.output_rate) / 1e6
        if usage_available
        and args.input_rate is not None
        and args.output_rate is not None
        else None
    )
    report = {
        "schema_version": 1,
        "evidence_type": "real_model_requests",
        "model": args.model,
        "model_role": args.role,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_sha256": identity,
        "prompt_version": PROMPT_VERSION,
        "metrics": score(rows, labels),
        "completed_requests": len(rows),
        "planned_requests": len(benchmark["examples"]),
        "status": "complete"
        if len(rows) == len(benchmark["examples"])
        else "incomplete",
        "request_interval_seconds": interval,
        "resumed_from_records_sha256": prior["records_sha256"] if resume else None,
        "duration_note": (
            "Sum of active run segments, excluding downtime between resumes"
        )
        if resume
        else "Single run, including request pacing",
        "duration_seconds": wall,
        "latency_median_seconds": statistics.median(
            r["duration_seconds"] for r in rows
        ),
        "latency_p95_seconds": sorted(r["duration_seconds"] for r in rows)[
            math.ceil(0.95 * len(rows)) - 1
        ],
        "throughput_clauses_per_second": len(rows) / wall,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "cost_per_1000_clauses_usd": cost * 1000 / len(rows)
        if cost is not None
        else None,
        "pricing": {
            "input_usd_per_million": args.input_rate,
            "output_usd_per_million": args.output_rate,
            "source": args.pricing_source,
        },
        "server_peak_memory_bytes": None,
        "server_memory_note": (
            "Measure on inference host; client RSS is not model memory."
        ),
        "records_sha256": hashlib.sha256(
            args.report.with_suffix(".jsonl").read_bytes()
        ).hexdigest(),
    }
    write_json(args.report, report)
    return report


def compare(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if any(r.get("status") == "incomplete" for r in reports):
        raise ValueError("Incomplete runs cannot be published as benchmark comparisons")
    if (
        not reports
        or len({r["benchmark_sha256"] for r in reports}) != 1
        or len({r["prompt_version"] for r in reports}) != 1
    ):
        raise ValueError("Only compare identical frozen benchmarks and prompts")
    identities = []
    for path, report in zip(paths, reports, strict=False):
        journal = path.with_suffix(".jsonl")
        if hashlib.sha256(journal.read_bytes()).hexdigest() != report["records_sha256"]:
            raise ValueError("Prediction journal checksum mismatch")
        rows = [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != report["metrics"]["sample_size"]:
            raise ValueError("Report/journal size mismatch")
        if score(rows, list(report["metrics"]["per_category"])) != report["metrics"]:
            raise ValueError("Reported metrics disagree with prediction journal")
        identities.append([(r["clause_id"], r["target"]) for r in rows])
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Prediction journals do not contain the same examples")
    public = {
        "schema_version": 1,
        "benchmark_sha256": reports[0]["benchmark_sha256"],
        "runs": reports,
    }
    write_json(output, public)
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("run")
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--role",
        choices=[
            "api-zero-shot",
            "frontier-zero-shot",
            "small-zero-shot",
            "fine-tuned",
            "quantized",
        ],
        required=True,
    )
    p.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between request starts; wall throughput includes pacing",
    )
    p.add_argument(
        "--stop-on-http-error",
        action="store_true",
        help="Preserve a partial journal and stop on provider HTTP errors",
    )
    p.add_argument(
        "--stop-on-access-error",
        action="store_true",
        help="Stop on auth/quota errors or three consecutive provider failures",
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Continue a verified incomplete report into a new file; "
            "retain all original failures"
        ),
    )
    p.add_argument("--key-env", default="MODEL_API_KEY")
    p.add_argument("--max-requests", type=int, default=200)
    p.add_argument("--input-rate", type=float)
    p.add_argument("--output-rate", type=float)
    p.add_argument("--pricing-source")
    p = sub.add_parser("compare")
    p.add_argument("--reports", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.data, args.output, args.count, args.seed)
    elif args.command == "run":
        result = run(args)
    else:
        result = compare(args.reports, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
