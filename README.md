# ClauseForge

**Status: In Development**

ClauseForge is a planned contract-clause intelligence system. The project is
intended to support clause classification, grounded risk analysis, model
evaluation, quantized deployment, and production-style inference. Risk
analysis and real deployment remain unimplemented. Phase 3A provides
training infrastructure, but no production transformer model.

## Current status

**Phase 3A training infrastructure and the serving/API foundation are complete.**

**The CPU-safe safety and robustness harness is complete.**

**The SEC EDGAR unlabeled OOD data pipeline is complete and offline-validated.**

**Phase 3B GPU feasibility is validated; the full Qwen rank ablation is pending.**

**A bounded Tesla T4 pilot mode is supported but has not been run here.**

**Quantization and production-serving preparation is implemented; no model has
been quantized or deployed.**

Final adapter training and rank selection, real quantization, deployment, and
final serving benchmarks remain pending.

The completed 10-step T4 pilot was a successful infrastructure/training pilot,
but produced 0 exact taxonomy-valid validation outputs. This indicates an
output-compliance problem requiring diagnosis before a larger GPU run; it is
not final model performance or a conclusion that the model failed. See
[Phase 3B output diagnostics](docs/phase3b_output_diagnostics.md).

The repository contains the Phase 0 foundation, Phase 1 CUAD data pipeline,
Phase 2 classical baselines/evaluation, and a Phase 3A training harness. The
harness builds deterministic exact-label examples, measures tokenizer
truncation, attaches LoRA adapters, configures optional QLoRA, records adapter
checkpoints and metadata, and bridges pre-trained generative classifiers into
the Phase 2 evaluator. A FastAPI service exposes health, readiness, and
versioned clause classification through an implementation-neutral provider.

ClauseForge does not yet have a production fine-tuned model and does not analyze
legal risk or provide a deployed inference service. Its local
API defaults to a clearly identified deterministic development stub. Real
full fine-tuned model evaluation remains pending GPU rank experiments.

Qwen2.5-7B-Instruct successfully completed a genuine 4-bit NF4 QLoRA optimizer
step on a free Colab Tesla T4. This proves infrastructure feasibility, not
model quality. Controlled rank-8/16/32/64 training remains to be run.

The official one-step sanity run also completed on T4. Full 11,223-example
training was manually interrupted due to free Colab runtime constraints, not
OOM or model failure. Pilot mode uses deterministic 512-example train and
256-example validation subsets, frequent resumable checkpoints, and never
evaluates held-out test. Pilot results are explicitly non-final.

## Architecture direction

The target system will separate data preparation, training, evaluation,
quantization, and serving concerns. Shared configuration and observability will
live in the core Python package. See [docs/architecture.md](docs/architecture.md)
for the planned component boundaries.

## Development roadmap

- [x] Phase 0: Repository foundation and development tooling
- [x] Phase 1: Public dataset ingestion and validation
- [x] Phase 2: Clause classification baseline and evaluation harness
- [x] Phase 3A: Reproducible LoRA/QLoRA training infrastructure
- [ ] Phase 3B: Real-model training and comparison
- [ ] Phase 3B Stage 1: Qwen rank ablation (GPU feasibility validated)
- [x] Serving/API foundation with development stub
- [x] SEC EDGAR out-of-distribution data pipeline
- [ ] Phase 3: Grounded risk analysis
- [ ] Phase 4: Evaluation and frontier-model comparison
- [x] Phase 5 preparation: quantization plans, manifests, providers, benchmarks
- [ ] Phase 5 execution: build and validate real artifacts and deploy serving
- [ ] Phase 6: Regression evaluation and production hardening

Only one phase is implemented and validated at a time. Later phases remain
planning targets, not claims of working functionality.

## Local inference API

```bash
uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port 8000
```

The default `mock-development` provider is deterministic test infrastructure,
not a trained legal model. Transformer mode requires explicit model, adapter,
and tokenizer paths; missing artifacts fail readiness without fallback.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"This agreement is governed by the laws of Delaware."}'
```

OpenAPI documentation is available at `/docs`. The service assists with
contract clause analysis and does not provide legal advice.

AWQ/GGUF preparation, vLLM/llama.cpp backends, deployment manifests, exact
caching, and benchmark methodology are documented in
[quantization](docs/quantization.md), [serving backends](docs/serving_backends.md),
and [benchmarking](docs/benchmarking.md). These are preparation capabilities,
not evidence that a model was quantized, evaluated, or deployed.

## Safety and robustness

The offline harness evaluates synthetic adversarial clauses, exact taxonomy
integrity, instruction-like text, malformed-input rejection, and deterministic
paraphrase consistency through the serving provider boundary.

```bash
python scripts/run_safety_eval.py --provider mock --output eval/results/safety
python scripts/run_safety_eval.py --provider classical --output eval/results/safety-rules
```

Results are ignored and labeled development evidence, not final model
performance. Final trained-model safety performance remains pending Phase 3B
GPU execution. See [the threat model](docs/safety_and_threat_model.md).

## Local development

Python 3.11 is the supported development and CI version.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make check
```

Configuration is read from environment variables. Copy `.env.example` only as
a reference; the application does not automatically load `.env` files.

## Preparing CUAD

Download CUAD v1 yourself from the public [Atticus Project CUAD
repository](https://github.com/TheAtticusProject/cuad). Dataset downloads are
never performed implicitly and must remain under an ignored data directory.

```bash
python -m clauseforge.data.prepare \
  --input data/raw/cuad/CUAD_v1.json \
  --output data/processed/cuad/1.0.0
```

Add `--dry-run` to validate and summarize the input without writing files. The
committed `scripts/prepare_cuad.py` wrapper provides the same command.

Outputs are deterministic JSON/JSONL where practical: `contracts.jsonl`,
`clauses.jsonl`, unlabeled `segments.jsonl`, `splits.json`, `manifest.json`, and
`statistics.json`. Contract, clause, and segment IDs are content-derived. The
manifest creation timestamp is run metadata; checksums cover the source and
deterministic generated files.

Splitting occurs at the contract level with a deterministic seed and default
80/10/10 ratios. This prevents annotations from one contract appearing in more
than one split. CUAD annotation text and offsets remain ground truth; the
pipeline does not invent, rewrite, or relocate labels.

## Baseline evaluation

Phase 2 evaluates four pre-fine-tuning baselines on the held-out CUAD test
split. Macro F1 is primary because all 41 categories should carry equal weight
despite strong class imbalance.

| Baseline | Test accuracy | Test macro F1 |
|---|---:|---:|
| Majority class | 0.180251 | 0.007450 |
| Keyword rules | 0.316614 | 0.201389 |
| TF-IDF logistic regression | 0.779781 | 0.672683 |
| TF-IDF LinearSVC | 0.786050 | 0.665255 |

These are measured classical baselines, not fine-tuned language-model results.
See [eval/README.md](eval/README.md) for methodology and limitations.

## Preparing SEC EDGAR OOD data

EDGAR retrieval is explicit and requires `SEC_USER_AGENT` and
`SEC_CONTACT_EMAIL`:

```bash
python scripts/fetch_edgar_contracts.py --output data/raw/edgar --cik 320193 --limit 25 --resume
python scripts/prepare_edgar.py --input data/raw/edgar/documents.jsonl --output data/processed/edgar/1.0.0
python scripts/validate_edgar.py --input data/processed/edgar/1.0.0
```

EDGAR candidates are unlabeled OOD inputs, never CUAD training examples.
Downloaded and processed files remain ignored. See
[docs/edgar_ood.md](docs/edgar_ood.md) for etiquette and limitations.

## Data and security

Use only public or explicitly approved synthetic data. Raw, interim, and
processed data artifacts are ignored by Git except for directory markers and
documentation. Never commit credentials, client information, proprietary data,
model weights, or generated evaluation outputs.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
