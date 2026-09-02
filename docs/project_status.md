# Project status

| Component | Status | Evidence | Remaining work |
|---|---|---|---|
| Data pipeline | COMPLETE | Deterministic CUAD ingestion, segmentation, validation, manifests | Rebuild only for a new source version |
| Classical baselines | VALIDATED | Held-out results and reproducible evaluation harness | Retain as fixed context |
| Training infrastructure | COMPLETE | LoRA/QLoRA configs, checkpoints, resume, pilot mode | Execute longer GPU experiments |
| GPU feasibility | VALIDATED | T4 load and real NF4 optimizer step | Obtain durable compute |
| T4 pilot | PARTIAL | 10-step run and checkpoint artifacts | Diagnose checkpoint-25 |
| Checkpoint diagnostics | VALIDATED | Checkpoint-10: 0/128 exact outputs; test untouched | Compare later checkpoint |
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

