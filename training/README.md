# Training

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
GPT-2 target `c_attn` is smoke-tested. Qwen2, Mistral, and Llama attention
targets remain experimental until their candidates are provisioned and tested.

## Real CUAD token analysis

The tracked `token_analysis_qwen2_5_7b.json` report uses the pinned
Qwen2.5-7B-Instruct tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28`.
It analyzes train (11,223 examples) and validation (1,324) only. At the planned
1,024-token limit, train min/median/p95/max are 62/120/251/837 and validation
are 62/126/251/617. No example is truncated in either split, so every category
has a zero truncation count and there is no disproportionately affected
category. The test split was not tokenized or used to choose the limit.
