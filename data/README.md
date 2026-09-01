# Data directories

No datasets are included in Phase 0.

- `raw/` is reserved for immutable source data.
- `interim/` is reserved for intermediate transformations.
- `processed/` is reserved for validated, model-ready artifacts.

The contents of these directories are ignored by Git. Only `.gitkeep` markers
are tracked. Use public or explicitly approved synthetic data, record its
provenance in a future manifest, and never place client or proprietary data in
the repository.
