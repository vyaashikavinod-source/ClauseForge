# Target architecture

This document distinguishes implemented foundations from planned system
components. Phase 1 implements local CUAD preparation, and Phase 2 implements
classical classification baselines and their evaluation harness. Production-scale
fine-tuning and risk evaluation remain planned. Quantization and production-serving
preparation are implemented; real artifact execution remains pending.

## Planned component boundaries

1. **Data preparation (partially implemented):** CUAD v1 SQuAD-style ingestion,
   normalization, validation, segmentation, splitting, manifests, and
   statistics are implemented. A separate SEC EDGAR pipeline creates unlabeled
   OOD candidates without entering CUAD splits.
2. **Classical baselines and evaluation (implemented):** expose a model-neutral
   classifier protocol, leakage-safe dataset views, metrics, contract-level
   bootstrap intervals, calibration diagnostics, confusion analysis, and
   machine-readable errors.
3. **Training (infrastructure implemented):** typed configurations, deterministic
   exact-label SFT examples, truncation analysis, LoRA/optional QLoRA setup,
   adapter checkpoints, metadata, and a local smoke path. Production-scale
   fine-tuning and explanations remain unimplemented.
4. **Extended evaluation (planned):** measure task quality, hallucination
   behavior, calibration, latency, and cost against versioned benchmark inputs.
5. **Quantization (preparation implemented):** typed AWQ/GGUF plans, strict
   merge compatibility, lineage manifests, checksums, and validation. No real
   artifact has been produced in this repository.
6. **Serving (foundation implemented):** FastAPI schemas, request identity,
   structured errors, privacy-preserving logs, liveness/readiness, and explicit
   mock, local-transformer, vLLM, and llama.cpp providers. Real adapter loading
   and deployment remain pending.
7. **Regression CI (partially implemented):** deterministic unit and fixture
   evaluation checks run in Phase 2 CI. Future model-dependent regression gates
   remain planned.

## Implemented foundation

The `src/clauseforge` package currently provides:

- environment-variable configuration with safe, secretless defaults;
- structured JSON logging based on the Python standard library; and
- package metadata and import boundaries for future modules.

The `src/clauseforge/data` package adds these Phase 1 boundaries:

- `cuad`: strict local parsing of the official SQuAD 2.0-style JSON;
- `models`: typed contracts, clauses, provenance, segments, and manifests;
- `normalize`: stable identifiers and authoritative span alignment;
- `segment`: numbered-section and heading detection with paragraph and sentence
  fallbacks; segmentation never uses fixed-size chunks;
- `validate`: record ownership, identifier, text, and offset invariants;
- `split`: seeded contract-level 80/10/10 assignment to prevent leakage;
- `manifest` and `statistics`: checksums, build metadata, and descriptive counts;
  and
- `prepare`: the fail-fast orchestration and command-line interface.

Annotated clauses come only from CUAD answers. Rule-based segments are
unlabeled candidate boundaries and do not replace or reinterpret CUAD ground
truth. Exact spans are preferred. Whitespace-equivalent alignment is allowed
only while anchored at the supplied start offset and is recorded in provenance;
the pipeline never searches for a different occurrence. Invalid annotations
stop preparation.

Phase-specific directories remain deliberately separate so later workflows do
not become coupled to notebooks or serving code. Data and generated artifacts
are excluded from version control by default.

The Phase 3B extension stays inside `clauseforge.training`: canonical taxonomy
metadata, pinned Qwen rank configs, assistant-only masking, non-reentrant
gradient checkpointing, adapter/resume checkpoints, resource metrics,
exact-output validation, and rank selection. Train and validation are the only
optimization and model-selection views; held-out test stays sealed.

## Design constraints

The release flow is: data → training → adapter → merge → quantization → artifact
validation → provider → exact cache → API → structured logging/metrics →
deployment. Data/training interfaces, adapter infrastructure, quantization
planning, artifact validation, provider/cache/API boundaries, and operational
instrumentation are implemented. Final adapter selection, merge, real
quantization, final artifact validation, model serving benchmarks, and deployment
have not been executed.

Release readiness is evaluated in four independent layers: infrastructure,
model artifact, deployment, and production validation. Infrastructure readiness
cannot promote blocked model or validation layers.


- Python 3.11 is the target runtime.
- Secrets must enter through the execution environment, never source control.
- Data provenance and model artifacts must be explicit and reproducible.
- Evaluation must distinguish measured results from plans or expectations.
- External services and paid APIs require explicit authorization.

Architectural decisions that materially change these constraints should be
recorded in `docs/decisions/` before implementation.

## SEC EDGAR OOD boundary

`clauseforge.data.edgar` owns explicit identified retrieval, tolerant HTML/text
extraction, explainable filtering, checksum deduplication, segmentation,
validation, manifests, and statistics. It reuses `segment_contract` while
preserving EDGAR provenance. `clauseforge.evaluation.ood` reports only
label-free validity and distributions. EDGAR data never enters CUAD splits.

## Phase 3A training boundary

`clauseforge.training` consumes the existing leakage-safe evaluation dataset.
It materializes train and validation examples, validates the 41-label taxonomy,
and asserts train, validation, and test contract sets are pairwise disjoint.
Test examples are never returned to the trainer.

PEFT supplies adapter attachment and reload. A local tiny GPT-2 proves the path
without downloading a checkpoint. Optional 4-bit configuration is isolated
behind CUDA and `bitsandbytes` checks. `TransformerClassifier` structurally
implements the Phase 2 protocol so future checkpoints use existing metrics.

## Serving boundary

`clauseforge.serving.app` owns HTTP transport and response formatting;
`providers.base` defines the model-independent asynchronous contract. The
deterministic `mock-development` provider supports local integration without
claiming model quality. The transformer provider is explicitly configured and
fails readiness when artifacts are absent; it never silently selects the mock.

Every successful response is checked against the packaged, versioned 41-label
CUAD taxonomy. Unknown output is a controlled error, never a fuzzy match.
Request middleware logs request ID, route, status, latency, and safe size
metadata without clause text. No CORS middleware is enabled by default.

Production preparation adds explicit vLLM and llama.cpp HTTP adapters without
silent mock fallback. Deployment manifests and exact-cache keys bind model,
adapter, taxonomy, prompt, quantization, and inference identities. Heavy
backend runtimes stay outside the base dependency set so offline CI remains
deterministic.

## Safety evaluation boundary

`clauseforge.safety` loads versioned synthetic fixtures and evaluates any
serving-compatible provider. It reuses the packaged taxonomy and exact output
validation. Adversarial and paraphrase artifacts go to ignored evaluation
directories. Mock and classical runs validate harness behavior only; real-model
safety measurement remains pending Phase 3B.
