# MODEL CARD — PRE-RELEASE / MODEL NOT FINALIZED

## Historical provisional RC0

RC0 was a provisional rank-8 validation-only candidate. Its corrected pilot
reported validation accuracy about 0.4766, macro F1 about 0.4612, weighted F1
about 0.4688, exact-ID rate about 0.8516, and invalid-output rate about 0.1484.
These are not final metrics or comparisons against the held-out classical test
baseline. Test, final safety/OOD, quantization, benchmark, and deployment are
pending.

## Phase 3B experiment families

Historical adapters target full canonical questions under
`cuad-canonical-question-v1`. Phase 3B v2 targets exact stable IDs under
`cuad-category-id-v1` and must use `cuad-classification-id-v2`. Its earlier
25-step T4 pilot produced accuracy and macro/weighted
F1 of 0, 0 exact IDs, and invalid-output rate 1.0. This is diagnostic evidence,
not final performance; the held-out test was untouched.

After corrected sequence construction, an eight-example training-only overfit
diagnostic reached 100% exact-ID memorization at step 20 and retained it through
step 100. The corrected 100-step validation pilot then produced the provisional
RC0 metrics above; larger/full training and final selection remain pending.

## Status and use

ClauseForge has no selected final language-model adapter. The intended future
use is assisting classification of contract clauses into the authoritative
41-category CUAD taxonomy. It is not legal advice and is out of scope for
autonomous legal decisions, contract approval, or unsupported risk conclusions.

## Data and experiments

Training infrastructure consumes public CUAD-derived train/validation examples
with contract-level isolation. Qwen2.5-7B-Instruct NF4 rank-8 feasibility was
validated on T4. A 25-step canonical-target pilot completed but produced 0
exact outputs across 128 validation examples (128 unrelated). This is
historical infrastructure evidence, not final performance.

## Evaluation and safety

Configuration selection uses validation only. The held-out test stays sealed
until a single model/configuration is locked, then is evaluated once. Final
safety and unlabeled SEC EDGAR OOD evaluations must follow. Exact taxonomy
validation, prompt-injection fixtures, sanitized serving errors, and the legal
disclaimer are mandatory boundaries.

## Limitations and pending evaluation

Corrected validation compliance is unresolved, rank selection is incomplete,
and no final test, quantized artifact, latency, or throughput evidence exists.
Training memorization is not generalization; performance claims must wait for
the release gates.
# Active candidate status

The serving architecture can load checkpoint 700 as an **active trained model
candidate — not final release**. Its category-ID output remains subject to exact
taxonomy validation. This is not a claim of held-out test, final safety/OOD,
production benchmark, or release performance.

**FINAL MODEL SELECTION: PENDING. HELD-OUT TEST: UNTOUCHED. FINAL RELEASE: BLOCKED.**
