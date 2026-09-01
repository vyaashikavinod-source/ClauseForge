# Scripts

`prepare_cuad.py` is a thin local entry point for the Phase 1 CUAD pipeline.

The EDGAR fetch, prepare, validate, and review-sample scripts are explicit
entry points for the isolated unlabeled OOD pipeline. Tests never call SEC.

`train_classifier.py` supports Phase 3B Qwen sanity, full, and resume modes.
`run_phase3b_rank_sweep.py` prints controlled rank commands and selects only
from measured validation macro-F1 inputs; it never launches training itself.

`merge_adapter.py` validates adapter/base compatibility and prints a merge
plan. `quantize_model.py` validates AWQ/GGUF configuration and prints a plan.
`validate_deployment_artifact.py` checks lineage and checksums. None converts a
model implicitly. `benchmark_serving.py` measures a separately running endpoint;
mock measurements are development-only, never model performance.
