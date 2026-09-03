# ClauseForge rank-8 RC0

> **PROVISIONAL MODEL CANDIDATE — VALIDATION ONLY — NOT FINAL RELEASE**

`clauseforge-qwen25-7b-r8-rc0` records the corrected Phase 3B v2 rank-8
candidate. It is metadata-only: no adapter binary is in this repository and
the manifest is not an activation instruction.

## Evidence available

The corrected `category_id` pipeline (`cuad-category-id-v1`, prompt
`cuad-classification-id-v2`) ran for 100 optimizer steps on a Tesla T4 using
256 deterministic training examples and 128 validation examples. It saw 1,600
training examples. Validation-only pilot metrics were approximately:

- accuracy: 0.4766
- macro F1: 0.4612
- weighted F1: 0.4688
- exact valid ID outputs: 109/128 (0.8516)
- invalid-output rate: 0.1484

These are **NOT FINAL MODEL PERFORMANCE**. The run is promising validation
evidence for the corrected pipeline, not evidence of production readiness.

## Explicitly incomplete

- NO HELD-OUT TEST EVALUATION
- NO FINAL SAFETY/OOD EVALUATION
- NO QUANTIZATION
- NO REAL SERVING BENCHMARK
- NO DEPLOYMENT

The example manifest in `configs/artifacts/` fails final-candidate and release
gates visibly. Its adapter path and checksum remain unset until an externally
stored adapter is imported and validated.
