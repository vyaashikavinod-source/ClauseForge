# Phase 3B output diagnostics

## Pilot evidence

The 10-step Tesla T4 pilot completed successfully as an infrastructure and
training pilot. It used Qwen2.5-7B-Instruct with rank-8 NF4 QLoRA, selected 256
training and 128 validation examples, saw 160 training examples, and created
`checkpoint-10`. Validation produced 0 exact taxonomy-valid outputs: 85 were
invalid and 43 malformed, for an invalid-output rate of 1.0. Held-out test was
not evaluated.

This is **PHASE 3B T4 PILOT — NOT FINAL MODEL PERFORMANCE**. It identifies an
output-compliance problem before a larger run. It does not show that the model
failed, complete Phase 3B, or establish a Phase 2 baseline comparison.

## Current generation and validation boundary

`render_prompt` produces versioned `System`, `User`, and `Assistant` sections.
The assistant target is the full authoritative CUAD question. Validation
tokenizes with truncation at the configured sequence length, greedily generates
at most `validation_max_new_tokens` (160 by default), slices away prompt token
IDs, and decodes only generated tokens with special tokens removed.

Acceptance remains strict: after outer whitespace stripping, output must equal
one of the 41 full canonical strings. Diagnostics recognize empty,
short-name-only, canonical-prefix, canonical-substring, commentary-wrapped,
truncated, malformed, and unrelated shapes, but do not accept them. No fuzzy
mapping is used.

Final validation writes ignored `validation_predictions.jsonl` records with
clause ID; canonical target, stable ID, and name; raw and normalized output;
strict status/reason; exact-match flag; and generated/target token counts.
Aggregates enter validation metrics. The 160-token cap is retained because
canonical targets can be long; diagnostics now show whether it is reached.

Future GPU-only checkpoint inspection is explicit and validation-only:

```bash
python scripts/evaluate_phase3b_checkpoint.py \
  --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --checkpoint checkpoints/phase3b/pilot/EXPERIMENT/checkpoint-10 \
  --output checkpoints/phase3b/pilot/EXPERIMENT/checkpoint-10-diagnostics \
  --pilot --pilot-validation-examples 128 --diagnostics
```

## Target representation recommendation

| Representation | Reliability and efficiency | Integrity/serving | Migration risk |
|---|---|---|---|
| Full canonical question | Hardest to reproduce; many tokens | Directly preserves current ground truth | None; current behavior |
| Stable category ID | Short, unique, ASCII, easiest to validate | Strong taxonomy key and clean serving contract | Moderate: new target/template/checkpoint version |
| Short category name | Short and readable | Display text is a weaker identifier | Moderate; renaming can break identity |
| Constrained decoding over 41 categories | Highest expected compliance | Preserves identity when driven by versioned taxonomy | Highest complexity; backend-specific validation |

Recommendation: retain the canonical question as authoritative metadata, but
prefer stable category IDs for a future deliberately versioned target migration.
Evaluate constrained decoding independently. Do not retrofit current checkpoints
or accept diagnostic short names as canonical outputs.

## Historical checkpoint compatibility

Validation-only evaluation uses persisted `experiment_config.json`,
`pilot_config.json`, `resume_state.json`, `adapter_metadata.json`, and the
dataset manifest as historical lineage. The stored experiment ID is reproduced
from the stored configuration rather than recalculated from current defaults.

Model-critical checks cover model/tokenizer name, immutable revision, family,
precision, quantization mode/type and double-quantization setting, LoRA rank and
alpha, target modules, and agreement with adapter metadata. These fields
determine which base weights, tokens, and adapter structure can be loaded.
Training-critical evaluation checks cover historical
experiment identity, pilot run mode, selected-subset checksum, prompt template,
maximum sequence length, and exact taxonomy. These bind the meaning and input
population of the checkpoint.

`validation_max_new_tokens` is evaluation-only: it controls post-training
generation length and cannot change adapter tensors or learned weights. A
current explicit value may override its historical value (or its absence), and
both values are recorded. Output paths, documentation tags, save/eval cadence,
and other optimizer controls are not used to decide validation-only loading;
they remain part of persisted lineage. Training resume continues to use the
stricter current experiment ID and subset checks and is unchanged.

Compatibility reports contain only named checks and safe configuration values;
they omit local paths and raw clause content. Blocking field names are included
in developer-facing errors so incompatibility can be diagnosed without exposing
private filesystem information.

Historical pilot evaluation also restores persisted selected-example IDs. The
pilot used 256 train and 128 validation examples, whose combined checksum is
`a287835bc6954916c2d9066e7a4b76b18c6946419290abf839f6996bc73ba458`.
Reconstructing with the later 512-train default instead produces
`d16b99769131b050fde76cd4a8016f6cf394d030b9e8e338664006e25ff9bbf6`;
that was the subset mismatch.

When present, `selected_train_examples.json` and
`selected_validation_examples.json` are authoritative. Their ordered IDs,
individual checksums, counts, category coverage, split identity, and zero-test
marker are validated against the current source dataset. If legacy selection
files are absent, reconstruction uses persisted pilot counts and seed, never
current defaults. The combined checksum must still equal both pilot and resume
metadata.

Omitting `--pilot-validation-examples`, or explicitly passing the persisted 128,
evaluates the exact historical validation IDs. A different explicit size creates
a deterministic new validation-only diagnostic sample. The report marks this as
`evaluation_override=true` and records historical training, historical
validation, and evaluation validation summaries separately; it never relabels
the new sample as the original pilot subset. Historical training IDs remain
lineage-only and are not passed through model evaluation.
