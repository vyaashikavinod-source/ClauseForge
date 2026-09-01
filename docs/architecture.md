# Target architecture

This document describes the intended ClauseForge architecture. Except for the
small shared configuration and logging foundation, the components below are
planned and are not implemented in Phase 0.

## Planned component boundaries

1. **Data preparation (planned):** ingest approved public or synthetic sources,
   validate provenance and schemas, and produce versioned model-ready data.
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
