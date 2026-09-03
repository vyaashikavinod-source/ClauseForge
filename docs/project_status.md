# Project status

## Current handoff status

- Engineering platform: substantially complete; final artifact handoff is ready.
- Current ML result: promising corrected rank-8 validation pilot, not final.
- Final model: pending GPU compute, locked selection, one-time test, final
  safety/OOD, quantization, benchmark, and deployment authorization.

The 25-step T4 canonical-target pilot completed as infrastructure evidence but
returned no exact canonical validation outputs. That family is historical. The
category-ID v2 25-step T4 pilot also completed: training loss 3.095277,
validation loss 1.416373, accuracy/macro F1/weighted F1 0, exact IDs 0, and
invalid-output rate 1.0 (3 commentary-wrapped, 8 malformed, 120 unrelated out
of 128; categories can overlap). Test remained sealed. Phase 3B is not complete.

The corrected train-only overfit diagnostic subsequently reached 8/8 exact IDs
at step 20 and stayed exact through step 100. The corrected 100-step pilot then
reached validation macro F1 about 0.4612 and 109/128 exact IDs. This is
provisional validation evidence, not final performance.

| Component | Status | Evidence | Remaining work |
|---|---|---|---|
| Data pipeline | COMPLETE | Deterministic CUAD ingestion, segmentation, validation, manifests | Rebuild only for a new source version |
| Classical baselines | VALIDATED | Held-out results and reproducible evaluation harness | Retain as fixed context |
| Training infrastructure | COMPLETE | LoRA/QLoRA configs, checkpoints, resume, pilot mode | Execute longer GPU experiments |
| GPU feasibility | VALIDATED | T4 load and real NF4 optimizer step | Obtain durable compute |
| T4 pilot | PROVISIONAL | Corrected 100-step validation pilot completed | Run larger/full rank-8 training |
| Checkpoint diagnostics | VALIDATED | Canonical pilot: 128/128 unrelated; test untouched | Validate category-ID outputs |
| Serving/API | COMPLETE | Strict FastAPI boundary and offline tests | Validate with final model |
| Safety harness | COMPLETE | Synthetic/offline robustness framework | Rerun on final model |
| EDGAR OOD | COMPLETE | Isolated unlabeled public OOD pipeline | Run final model OOD analysis |
| Quantization preparation | COMPLETE | AWQ/GGUF plans and manifests | Execute on selected adapter |
| Exact caching | COMPLETE | Version-isolated optional cache | Measure only with real serving |
| Benchmarking | COMPLETE | Honest async harness | Run final backend benchmark |
| Docker | COMPLETE | CPU mock Dockerfile/Compose | Execute smoke with running daemon |
| CI/CD | COMPLETE | Quality, security, container, gated release workflows | Observe hosted runs |
| Operations | COMPLETE | Readiness, logs, metrics, runbook | Environment-specific deployment review |
| Final adapter | BLOCKED | Rank selection is not complete | Select after validation-only experiments |
| Rank selection | BLOCKED | Checkpoint-25 diagnostics pending | Complete controlled GPU runs |
| Held-out test | BLOCKED | Deliberately sealed | Evaluate once after lock |
| Final quantization | PENDING | No final adapter | Merge, quantize, validate |
| Final deployment | PENDING | No final artifact/benchmarks | Deploy only after all release gates |

Repository structure: `src/clauseforge/` contains the package; `scripts/` holds
explicit CLIs; `training/` contains versioned experiment configuration; `tests/`
is CPU/offline; `demo/` is synthetic; `docs/` is the handoff set; generated
data, evaluations, checkpoints, and artifacts remain ignored.
