# Phase 3B experiment plan

Full rank-ablation results remain pending. Qwen2.5-7B-Instruct completed one
genuine 4-bit NF4 rank-8 QLoRA optimizer step on a free Colab Tesla T4 without
OOM. This is infrastructure feasibility evidence, not model performance.

## Selection objective

Validation macro F1 selects the model, rank, and configuration. The held-out
test set remains sealed until those choices are locked. Final reporting reuses
the Phase 2 evaluator and its contract-bootstrap intervals.

Current classical hurdle:

- TF-IDF Logistic Regression macro F1: **0.672683**
- 95% contract-bootstrap CI: **[0.608234, 0.689092]**

A fine-tuned model is not an improvement merely because its point estimate is
slightly higher. Compare interval uncertainty, latency, memory, operational
complexity, and error concentration as well.

## Candidate screen

| Candidate | Suitability and deployment implications |
| --- | --- |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.61B causal instruction model with grouped-query attention. The official card reports Apache-2.0 and a 32,768-token default configuration (131,072 with documented changes). Its strong instruction boundary makes exact-label adherence worth testing. NF4 QLoRA is architecturally compatible and is expected to fit a modern 16–24 GB CUDA GPU, subject to measurement; Apache terms simplify deployment review. |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7B, 32,768-context instruction model with Apache-2.0 licensing and standard attention projection names. Its compact, widely supported architecture provides a useful deployment-oriented counterpoint to Qwen. NF4 QLoRA is architecturally compatible; expect a 16–24 GB workstation envelope and broad inference-engine support. |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B, 128K-context instruction model. It offers a valuable architecture comparison, but access is gated and the custom Llama 3.1 license needs separate review. NF4 QLoRA is architecturally compatible; start validation on a 24 GB CUDA host and exclude it if access or license obligations outweigh its deployment value. |

Popularity is not a selection criterion. First run tokenizer analysis and a
20-example overfit check. Reject candidates that cannot preserve exact labels,
cannot be reproduced from a pinned revision, or exceed the compute budget.

## Controlled matrix

For each admitted candidate, hold split, template v1, seed 42, sequence length,
optimizer, schedule, epoch budget, and selection procedure fixed. Run LoRA ranks
**8, 16, 32, and 64**. Start with `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
expanding into MLP projections is a separate predeclared ablation.

1. Dry-run all candidate/rank configurations.
2. Run one short integration experiment per candidate.
3. Execute the rank matrix with identical validation cadence.
4. Select by validation macro F1; use weighted and rare-class F1 diagnostically.
5. Lock configuration and decoding.
6. Evaluate test once and compare its contract-bootstrap interval and practical
   costs with the classical hurdle.

Additional seeds are required around the leader before a final claim. Smoke
losses and smoke accuracy are excluded from all selection tables.

Stage 1 uses FP16 and the validated attention-only targets. Expected trainable
parameters are 5,046,272 (r8), 10,092,544 (r16), 20,185,088 (r32), and
40,370,176 (r64). Validation macro F1 selects the winner; exact ties select the
smaller rank. The Phase 2 held-out test macro F1 of 0.672683, with 95%
contract-bootstrap CI [0.608234, 0.689092], is context only and cannot be
directly compared with Qwen validation. Test stays sealed until configuration
lock.

## Bounded T4 pilot

Before the longer-lived rank sweep, rank 8 may run a 512-train/256-validation
pilot with seed 42 and five-step checkpoints. It reports genuine pilot training
and validation loss, accuracy, macro/weighted F1, invalid-output rate, resource
use, adapter size, and category coverage. These validation pilot measurements
remain separate from final rank selection and the Phase 2 held-out test.

Train supports 41 categories; validation supports 40. The pilot cannot create
coverage for a category absent from validation. Full rank ablation still
requires longer-lived GPU compute.
