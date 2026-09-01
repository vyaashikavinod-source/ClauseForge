# Training

The completed 10-step T4 pilot produced no exact taxonomy-valid validation
outputs. This is an output-compliance diagnostic finding, not final model
performance or a model-failure conclusion. See
[`docs/phase3b_output_diagnostics.md`](../docs/phase3b_output_diagnostics.md).

Phase 3A implements a reproducible supervised fine-tuning harness for exact
CUAD category classification. It is training infrastructure, not a trained
production model and not evidence that a transformer beats the Phase 2
classical baseline.

The versioned `cuad-classification-v1` template renders a system instruction,
the clause as the user message, and one unmodified authoritative category as
the assistant target. Dataset construction materializes train and validation
only. The test split is consulted solely to assert contract-ID disjointness.

Run the offline validation path:

```powershell
python scripts/train_classifier.py --config training/configs/smoke.yaml `
  --data data/processed/cuad/1.0.0-run-a --dry-run
```

Remove `--dry-run` to run the deliberately tiny, locally constructed GPT-2
smoke model. It downloads no weights. Output is written below the ignored
`checkpoints/` directory with metadata, tokenizer files, and PEFT adapter files.

SMOKE TEST — NOT MODEL PERFORMANCE. Its losses demonstrate executable plumbing
only and must not be compared with benchmark metrics.

Standard LoRA is supported through PEFT. The QLoRA example uses NF4 4-bit
loading and double quantization, but is not run in Phase 3A. It checks CUDA
before importing optional `bitsandbytes` and reports a clear error on CPU.
GPT-2 target `c_attn` is smoke-tested. Qwen2.5 attention targets `q_proj`,
`k_proj`, `v_proj`, and `o_proj` completed a genuine rank-8 4-bit QLoRA
optimizer step on a Tesla T4. This is GPU feasibility evidence, not model
quality. Mistral and Llama remain outside Stage 1.

## Phase 3B Stage 1: Qwen rank experiment

The authoritative taxonomy remains the 41 complete CUAD question strings.
`clauseforge.taxonomy` also exposes a stable ID and quoted short name; neither
replaces the canonical training target or serving value.

All rank configs use pinned Qwen2.5-7B-Instruct, NF4 double quantization, FP16,
length 1,024, micro-batch 1, accumulation 16, cosine scheduling, paged 8-bit
AdamW, seed 42, disabled cache, and non-reentrant gradient checkpointing. Only
rank and its 2× alpha change. Attention-only targets stay within the validated
T4 envelope; MLP projections are deferred.

```bash
# Offline config validation
python scripts/train_classifier.py --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml --data data/processed/cuad/1.0.0-run-a --dry-run
# One genuine GPU optimizer step
python scripts/train_classifier.py --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml --data data/processed/cuad/1.0.0-run-a --sanity-steps 1
# Full rank-8 run
python scripts/train_classifier.py --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml --data data/processed/cuad/1.0.0-run-a
# Resume
python scripts/train_classifier.py --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml --data data/processed/cuad/1.0.0-run-a --resume-from-checkpoint checkpoints/phase3b/<experiment>/checkpoint-<step>
# Print rank-8/16/32/64 commands
python scripts/run_phase3b_rank_sweep.py --data data/processed/cuad/1.0.0-run-a
```

Prompt tokens use the `-100` loss mask; only the assistant canonical answer
contributes to loss. Training uses train, selection uses validation macro F1,
and test remains sealed. Sanity output is labeled **GPU SANITY RUN — NOT MODEL
PERFORMANCE**.

## Tesla T4 pilot

GPU feasibility and the official one-step sanity command succeeded on a free
Colab Tesla T4. A full 11,223-example run was manually interrupted because the
short-lived runtime was too slow; there was no OOM or model failure, and the run
must not be described as complete.

```bash
python scripts/train_classifier.py \
  --config training/configs/phase3b/qwen25_7b_qlora_r8.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --pilot
```

The default pilot uses 512 deterministic stratified train examples and 256
validation examples. Optional sample limits and `--max-steps` bound runtime.
Checkpoints are written every five optimizer steps beneath
`checkpoints/phase3b/pilot/<experiment-id>/`; resume validates configuration,
mode, and selected-example checksum.

Train covers all 41 categories. Validation contains only 40 supported
categories, so the pilot covers all 40 and records the limitation. Test
contributes zero examples and metrics. Every artifact is labeled **PHASE 3B T4
PILOT — NOT FINAL MODEL PERFORMANCE**.

## Real CUAD token analysis

The tracked `token_analysis_qwen2_5_7b.json` report uses the pinned
Qwen2.5-7B-Instruct tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28`.
It analyzes train (11,223 examples) and validation (1,324) only. At the planned
1,024-token limit, train min/median/p95/max are 62/120/251/837 and validation
are 62/126/251/617. No example is truncated in either split, so every category
has a zero truncation count and there is no disproportionately affected
category. The test split was not tokenized or used to choose the limit.
