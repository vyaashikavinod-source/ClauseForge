# Target architecture

This document distinguishes implemented foundations from planned system
components. Phase 1 implements local CUAD preparation only; model training,
model evaluation, quantization, and inference remain planned.

## Planned component boundaries

1. **Data preparation (partially implemented):** CUAD v1 SQuAD-style ingestion,
   normalization, validation, segmentation, splitting, manifests, and
   statistics are implemented. Other sources are not implemented.
2. **Training (planned):** build reproducible clause-classification and grounded
   explanation experiments with explicit configurations and tracked artifacts.
3. **Evaluation (planned):** measure task quality, hallucination behavior,
   calibration, latency, and cost against versioned benchmark inputs.
4. **Quantization (planned):** create and validate deployment-oriented model
   variants without changing evaluation contracts.
5. **Serving (planned):** expose versioned inference interfaces with input
   validation, observability, and clear failure behavior.
6. **Regression CI (planned):** run suitably scoped deterministic checks before
   changes are accepted. Model-dependent evaluation is not part of Phase 0 CI.

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

## Design constraints

- Python 3.11 is the target runtime.
- Secrets must enter through the execution environment, never source control.
- Data provenance and model artifacts must be explicit and reproducible.
- Evaluation must distinguish measured results from plans or expectations.
- External services and paid APIs require explicit authorization.

Architectural decisions that materially change these constraints should be
recorded in `docs/decisions/` before implementation.

SEC EDGAR remains a planned out-of-distribution evaluation source. No EDGAR
ingestion or placeholder implementation exists in Phase 1.
