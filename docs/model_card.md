# MODEL CARD — PRE-RELEASE / MODEL NOT FINALIZED

## Status and use

ClauseForge has no selected final language-model adapter. The intended future
use is assisting classification of contract clauses into the authoritative
41-category CUAD taxonomy. It is not legal advice and is out of scope for
autonomous legal decisions, contract approval, or unsupported risk conclusions.

## Data and experiments

Training infrastructure consumes public CUAD-derived train/validation examples
with contract-level isolation. Qwen2.5-7B-Instruct NF4 rank-8 feasibility was
validated on T4. A 10-step pilot completed, but checkpoint-10 produced 0 exact
canonical outputs across 128 validation examples (2 short-name-only, 122
unrelated, 4 malformed). This is diagnostic pilot evidence, not final model
performance or a model-failure conclusion.

## Evaluation and safety

Configuration selection uses validation only. The held-out test stays sealed
until a single model/configuration is locked, then is evaluated once. Final
safety and unlabeled SEC EDGAR OOD evaluations must follow. Exact taxonomy
validation, prompt-injection fixtures, sanitized serving errors, and the legal
disclaimer are mandatory boundaries.

## Limitations and pending evaluation

Output compliance is unresolved, rank selection is incomplete, checkpoint-25
diagnosis awaits GPU access, and no final test, quantized artifact, latency, or
throughput evidence exists. Performance claims must wait for the release gates.

