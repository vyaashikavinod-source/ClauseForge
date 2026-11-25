# Evidence-first portfolio experiments

## What actually ran

On 2026-11-17 the official public CUAD archive was downloaded from
`https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip`.
The existing parser prepared 510 contracts and 13,823 authoritative annotated
spans. No label was inferred from an entire contract or an unanswered question.

The new harness froze 200 unique test clauses, sampled round-robin across
categories. Contracts are disjoint from training and validation. Normalized
exact-text duplicates appearing in train/validation are excluded, and ambiguous
multi-label span text is excluded from this single-label benchmark. This is
exact-text decontamination, not proof against paraphrase leakage or foundation
model pretraining contamination. Forty of the 41 categories are represented.
The unsupported category remains visible with zero support and no within-class
accuracy. All-category and supported-category macro F1 are reported separately.

Benchmark checksum:
`087c64082dc9ca2089d272650d5072000692222451a39057a083e541eaceb8d8`.

The frozen file is `eval/outputs/portfolio/benchmark-200.json`; raw/processed
CUAD and evaluation outputs stay ignored. This is a separately versioned
portfolio benchmark and does not silently unlock the existing final-release
held-out gate. Do not tune prompts, hyperparameters, or selection against it.
Use validation data for iteration, then evaluate each locked candidate once.

## Measured public-model baseline

A real local zero-shot run on the frozen 200 clauses has now completed using the
publisher's Qwen2.5-1.5B-Instruct-GGUF Q4_K_M model and official llama.cpp b12494.
It is a public instruction model, not our fine-tuned adapter or a frontier model.

| Measurement | Recorded value |
|---|---:|
| Accuracy | 20.0% (40/200) |
| Macro F1 across all 41 categories | 0.191002 |
| Macro F1 across 40 supported categories | 0.195777 |
| Invalid exact-ID outputs | 57.0% |
| Request errors | 0/200 |
| Median / p95 latency | 1.3486 s / 2.7193 s |
| Sequential throughput | 0.64494 clauses/s |
| Total benchmark wall time | 310.1046 s |
| Input / output tokens | 516,120 / 1,365 |
| API token charges / per 1k clauses | $0 / $0 |
| Shared inference-process peak working set | 2,080,186,368 bytes |

The memory value is the Windows peak of the actual inference process shared by
the incident and clause experiments, not isolated model-only or per-request memory.
API cost excludes hardware, electricity, training and hosting. Exact-ID failures
were counted as incorrect, not repaired or discarded. The public report is generated
from the checksummed prediction journal with the common scorer.

Model publisher revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, SHA256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
The model was served on loopback only and stopped after measurement. This tests
local quantized serving; it is not a quantization of our untrained adapter.

**Frontier API comparison and public model hosting remain unexecuted.** A real free-Colab QLoRA pilot and held-out evaluation have now completed; see the recorded pilot below. GGUF export and loopback quantized serving have completed; public model hosting is not provisioned.

## Reproduce the preparation

Use Python 3.11/3.12 for the full model stack. The data/evaluation-only commands
also run on the local Python 3.13 interpreter with PyYAML installed.

```powershell
$env:PYTHONPATH = 'src'
python scripts/prepare_cuad.py --input data/raw/cuad/CUADv1.json --output data/processed/cuad/portfolio-v1
python -m clauseforge.evaluation.portfolio prepare --data data/processed/cuad/portfolio-v1 --output eval/outputs/portfolio/benchmark-200.json --count 200
```

Preparation refuses to overwrite an existing frozen benchmark.

## Same evaluation for every model

Use a securely configured `MODEL_API_KEY` for an external endpoint. Never put
that key in the browser, source, evidence JSON, or a public demo. Use a fresh
report filename for every run. No provider is invoked by preparation or tests.
There is a hard request-count guard, no automatic retries, and every failed
request stays in the denominator as an invalid prediction. Provider failure
details record only the exception class, not credentials or HTTP bodies.

```powershell
python -m clauseforge.evaluation.portfolio run --benchmark eval/outputs/portfolio/benchmark-200.json --report eval/outputs/portfolio/frontier.json --endpoint https://api.openai.com/v1/responses --model YOUR_FRONTIER_MODEL --role frontier-zero-shot --max-requests 200
python -m clauseforge.evaluation.portfolio run --benchmark eval/outputs/portfolio/benchmark-200.json --report eval/outputs/portfolio/adapter.json --endpoint http://127.0.0.1:8080/v1/chat/completions --model YOUR_ADAPTER_MODEL --role fine-tuned
python -m clauseforge.evaluation.portfolio run --benchmark eval/outputs/portfolio/benchmark-200.json --report eval/outputs/portfolio/quantized.json --endpoint http://127.0.0.1:8080/v1/chat/completions --model YOUR_GGUF_MODEL --role quantized
python -m clauseforge.evaluation.portfolio compare --reports eval/outputs/portfolio/frontier.json eval/outputs/portfolio/adapter.json eval/outputs/portfolio/quantized.json --output src/clauseforge/frontend/evaluation.json
```

The same versioned taxonomy-aware prompt (`cuad-classification-id-v3-taxonomy`)
and exact category-ID scorer are used for every backend. A local OpenAI-compatible
server needs no API key. HTTPS is required for external endpoints. The comparison
rejects changed benchmark/prompt identities, mismatched journals, and metrics that
disagree with the recorded predictions. Output JSONL contains prediction IDs,
labels, model output, usage and measured time; it excludes contract text.

Provide current provider rates with `--input-rate`, `--output-rate` (USD per
million tokens), and `--pricing-source` to calculate an estimated API cost and
cost per 1,000 clauses. Missing usage/pricing remains null, never an invented zero.
The client records sequential throughput, p50/p95 latency, and wall time. Model
peak memory must be measured on the inference host; client RSS is not model RAM.
Local electricity, hardware, training and hosting costs are not API token costs.

## Small-model training and serving

`training/configs/portfolio_qwen15b_qlora.yaml` reuses the existing QLoRA trainer
with Qwen2.5-1.5B-Instruct at immutable revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, rank 8, NF4, and the same benchmark
prompt. Use the same processed train/validation data, keep test sealed, run a
short train/validation pilot first, and inspect truncation before a full run.
The taxonomy-bearing prompt requires a larger context than the old v2 prompt.

```powershell
python scripts/train_classifier.py --config training/configs/portfolio_qwen15b_qlora.yaml --data data/processed/cuad/portfolio-v1 --dry-run
```

Use the existing artifact registry, adapter merge, quantization and release
validation scripts after an actual trained candidate exists. Do not manufacture
an artifact manifest to make readiness green. Run the identical benchmark against
the adapter and GGUF server; export alone is not evidence of serving quality.

Render stays a lightweight frontend/API. The existing vLLM provider can call an
external server; `CLAUSEFORGE_REMOTE_API_KEY` now adds an Authorization header
only over HTTPS. Configure the backend URL/model identity and the artifact and
target-version metadata honestly. No remote inference host was provisioned.
Free GPU/host availability is account-dependent; no paid compute was started.

## Publishing

The local hero now contains an evidence table instead of decorative stats. It
shows pending states until validated results are exported. Mock labels remain
while the actual backend remains mock. Removing them now would misrepresent it.

The existing Render Docker demo still runs the previous GitHub commit. Publishing
these source edits requires write access to `vyaashikavinod-source/ClauseForge`.
Local Git authenticates as an account without repository write permission.
After source publication, manually redeploy the existing service. It must not be
reported as updated before its new deployment is live and checked.


## Recorded free-Colab pilot (November 17, 2026 Pacific)

A free Tesla T4 completed the pinned Qwen 1.5B QLoRA pilot: 64 unique train
clauses, 400 example passes, 25 optimizer steps, 1,433.270 seconds. Validation
used 32 clauses: 31.25% accuracy, macro F1 0.199187. This is a pilot, not a final
release. The selected adapter weights were exported and SHA256 verified locally.

The locked pilot then ran the exact original 200-clause held-out harness once:
42.5% accuracy (85/200), macro F1 across 41 categories 0.366466, supported-40
macro F1 0.375627, invalid outputs 14.5%, request errors 0%. Median/p95 latency
0.701914/0.891779 seconds, throughput 1.373968 clauses/s, duration 145.563801 s.
Usage: 510,320 input and 970 output tokens; $0 API charge and $0 Colab spend.
Whole-GPU inference snapshot: 3,583 MiB, not peak or isolated model memory.
Training peak allocated VRAM: 9,219,181,056 bytes. Hardware/precision differs
from the CPU/Q4 base run; do not interpret latency differences as a controlled
fine-tuning speedup. The base comparison is a public small model, not frontier.

Report: `eval/outputs/portfolio/colab-pilot-adapter.json`; exact journal has
SHA256 `a39ac05fa9bedf050645e159ef7fc44ee9c094386bad2b732e25cd762deb8793`.
Shared `compare` verified both journals, metrics, prompt and example identities.
Adapter safetensors SHA256:
`2a93cec21c4b2695108514f709fe5c298b64a27606900bfce98d5c4721aed211`.

Colab CUDA 13 required bitsandbytes 0.49.0. Keep Colab's CUDA-compatible torch,
use the smaller notebook dependency stack, and record `pip freeze`. The repository
full development lock did not install in that runtime. Actual dependency versions
and complete training diagnostics are preserved with the exported adapter.


## Quantized serving evidence (November 18, 2026 Pacific)

The merged Colab Q4_K_M export completed: 986,048,192 bytes, SHA256
`dbfedeeedf8a145f56b3c3f86db5980ac39d85ce4a9d6e10bc06a04c3254c515`.
The Colab runtime was lost before its full serving evaluation was verified.
The incomplete two-thread CPU trial, including its first request timeout, is
retained separately; it is not a completed quality score.

A durable portable deployment was then exported locally using the official
llama.cpp converter at commit `307e3f1551c2c3c6c0560c454ab46bb64baeb2df`:
the published Q4_K_M base plus the actual trained F16 GGUF LoRA adapter. The
adapter is 4,373,888 bytes, SHA256
`e60c991d247dd2faaca8e1142a4ba34301be2972295c8ebe472b20c871aece83`.
This deployment completed all 200 frozen requests: accuracy 41.0% (82/200),
all-41 macro F1 0.3860103, supported-40 macro F1 0.3956606, invalid IDs 13%,
request errors 0%. Median/p95 1.081513/1.931100 seconds; maximum 221.321189
seconds. Two requests exceeded 10 seconds; none were trimmed. Total time
463.836588 seconds, throughput 0.431186 clauses/s. Usage 510,320 input + 944
output tokens. Windows server peak working set 1,889,509,376 bytes, including
loading and a synthetic non-test warmup. The loopback server stopped afterwards.

These serving numbers apply to Q4 base + F16 LoRA, not the separately merged
986 MB file. Hardware, precision, chat formatting, prefix caching and host
conditions differ across measured backends, so timings are descriptive rather
than a controlled speedup claim. All charges were $0 API / $0 cloud GPU; local
hardware, electricity and network costs are not included.

Report/journal: `eval/outputs/portfolio/local-q4-lora.json` and `.jsonl`.
Journal SHA256: `79e602f929a386ae40030710fe112d5bc0c95f08941f14ced7cf2045b99f3b70`.
The shared compare command verified all three full runs, metrics, checksums and
example identities. Raw journals and the incomplete infrastructure trial are
published at the evidence page. No latest-frontier API comparison or public live model hosting is claimed. The ClauseForge interface backend remains honestly
labelled mock; source is published with the recorded evidence.


## Completed free API baseline

Gemini 3.5 Flash-Lite attempted all 200 frozen clauses with the identical prompt
and scorer: **153/200 correct (76.5%), all-41 macro F1 0.7360508732**.
Fourteen HTTP 503 responses are retained as incorrect (7% request-error rate).
Execution resumed from verified prefixes without retrying any failed clause.
This is a Flash-Lite API baseline, not a latest-frontier comparison.

Median/p95 request latency: 2.031631 / 9.859076 seconds. Active execution took
1,184.675244 seconds across four segments; pauses between segments are excluded.
Throughput includes 4.5-second request pacing to respect the free quota.
Successful responses report token usage, but full totals and usage-based cost
remain unknown because failed calls omitted usage. Billing stayed disabled;
no API charges or paid GPU services were used. API server memory is unavailable.

The shared comparison verified all 800 predictions across four completed runs:
matching example identities, benchmark/prompt checksums, journal hashes and
recomputed metrics. Raw journals, including earlier infrastructure failures,
are in `docs/evidence/clauseforge-predictions.json`. Public live hosting was
declined in favor of replay; the public interface retains its mock backend.

The reproducible free-Colab handoff is `notebooks/portfolio-free-colab.ipynb`.
This checked-in notebook is a handoff, not an executed-results artifact. Actual
training records are in `docs/evidence/ClauseForge-Colab-pilot-evidence.json`.
Frozen public CUAD test spans are in `eval/fixtures/cuad-portfolio-200.json`;
retain CUAD attribution and its source license when redistributing.


## Public replay and interpretation update — 21 November 2026

Five exact UI samples were run through the same Qwen2.5-1.5B Q4 base plus
the trained F16 GGUF LoRA on local CPU. Raw outputs, token usage, latency and
base/adapter hashes are in `src/clauseforge/frontend/sample-replays.json`.
The Non-compete sample returns `non_solicit_of_customers`, an incorrect
invalid ID (the taxonomy uses `no_solicit_of_customers`) that remains visible. Arbitrary input is refused, not sent to
the mock classifier. No public model inference is claimed.

“Trained” refers to the 25-step pilot on 64 unique clauses / 400 example
passes. The FP16 row and quantized-serving row use the same adapter.
42.5% versus 41.0% differs by three test clauses; no meaningful difference or
quantization improvement is established. Gemini's 35.5-point advantage over
quantized serving motivates future 3B/7B base, longer training and constrained
41-label decoding experiments, all explicitly unrun.

Known Gemini usage: 492,297 input and 807 output tokens. At Google standard
text rates checked 21 November 2026 ($0.30/$2.50 per million input/output),
this is $0.1497066, or $0.748533 per 1,000 attempted clauses. It is a paid-rate
equivalent lower bound because fourteen failed calls omitted usage. Actual
free-tier charge remains zero. Invalid outputs are shown separately from
request errors in the UI. Pricing source: https://aistudio.google.com/docs/pricing

## Constrained-output follow-up — 23 November 2026

The same 25-step adapter on the same Q4 base was evaluated with a 41-label
GBNF decoding constraint and an unchanged prompt. All 200 requests completed:
87/200 correct (43.5%), all-category macro F1 0.394628, zero invalid IDs and
zero request errors. Median/p95 latency: 1.276016 / 2.180212 seconds; sequential
throughput 0.745291 clauses/second; 510,320 input and 934 output tokens.
Windows process peak working set including loading/warmup: 1,904,082,944 bytes.
Local CPU, four threads, official llama.cpp b12494; no API charge is implied
to mean no hardware or electricity cost.

Paired comparison to the unconstrained Q4 adapter: five gained, zero lost,
exact two-sided McNemar p=0.0625. The reliable observed change is removal of
invalid label strings, not established general quality improvement. This
reuses a previously evaluated benchmark; it is a diagnostic follow-up, not
a new untouched final holdout. Raw outputs and checksums are preserved in
`docs/evidence/clauseforge-predictions.json`; paired counts are in
`docs/evidence/constrained-paired-comparison.json`.

A 3B zero-shot base comparison and a longer free-T4 training experiment are
in progress. The first 4096-token training attempt encountered GPU memory
pressure; the retry uses a 3072-token cap, 256 unique training clauses and
100 planned optimizer steps. No result from unfinished training is claimed.

The pinned Qwen2.5-3B Q4 base with constrained decoding completed 200 requests: 44.5% accuracy, 0.416103 macro F1, 0.0% invalid outputs. Median latency 2.049s. This is zero-shot base inference, not a trained 3B adapter.
