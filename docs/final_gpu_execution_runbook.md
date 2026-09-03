# Final GPU execution runbook

This is the authoritative future procedure. Commands are examples with
`<UPPER_CASE_PLACEHOLDERS>` replaced by operator-reviewed runtime paths or IDs.
Do not run test, safety, OOD, merge, quantization, benchmark, or deployment
before its preceding gate passes.

## A–E. Fresh private Colab checkout

In a Python cell, authenticate without echoing the GitHub token:

```python
import getpass, subprocess
token = getpass.getpass("GitHub token (repo read access): ")
subprocess.run(["gh", "auth", "login", "--with-token"], input=token,
               text=True, check=True, capture_output=True)
del token
subprocess.run(["gh", "repo", "clone", "ruwactrl/ClauseForge",
                "/content/ClauseForge"], check=True)
```

Then run:

```bash
set -euo pipefail
cd /content/ClauseForge
git fetch origin main
git checkout main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "<RUNBOOK_COMMIT>"
curl -LsSf https://astral.sh/uv/install.sh | sh
$HOME/.local/bin/uv venv --python 3.12 /content/ClauseForge/.venv
/content/ClauseForge/.venv/bin/python -m pip install -r requirements.txt
/content/ClauseForge/.venv/bin/python -m pip install bitsandbytes==0.49.0
/content/ClauseForge/.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
PY
nvidia-smi
```

Stop unless the verified commit is the commit approved for the run. Never put a
token in a clone URL, notebook output, archive, manifest, or shell history.

## F–G. Restore and deterministically prepare CUAD

Place the operator-supplied `CUAD_v1.json` at
`/content/ClauseForge/data/raw/cuad/CUAD_v1.json`, then:

```bash
cd /content/ClauseForge
.venv/bin/python scripts/prepare_cuad.py \
  --input data/raw/cuad/CUAD_v1.json \
  --output data/processed/cuad/1.0.0-run-a \
  --seed 42 --train-ratio 0.8 --validation-ratio 0.1 --test-ratio 0.1
test -f data/processed/cuad/1.0.0-run-a/contracts.jsonl
test "$(wc -l < data/processed/cuad/1.0.0-run-a/contracts.jsonl)" -eq 510
test "$(wc -l < data/processed/cuad/1.0.0-run-a/clauses.jsonl)" -eq 13823
```

Preparation creates the sealed test split, but training and model selection must
not evaluate or inspect it.

## H–K. Restore, train, resume, and validate rank 8

Optional restore from operator-provided local storage:

```bash
.venv/bin/python scripts/restore_gpu_artifacts.py \
  --archive <BACKUP_ARCHIVE> \
  --output checkpoints/restored/<EXPERIMENT_ID> \
  --expected-experiment-id <EXPERIMENT_ID>
```

Start full corrected rank-8 training:

```bash
.venv/bin/python scripts/train_classifier.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml \
  --data data/processed/cuad/1.0.0-run-a
```

Find the latest numeric checkpoint, inspect its lineage, then resume explicitly:

```bash
find checkpoints/phase3b_v2/full -type d -name 'checkpoint-*' -print | sort -V | tail -1
.venv/bin/python scripts/evaluate_phase3b_checkpoint.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --checkpoint <EXACT_CHECKPOINT_PATH> \
  --output <VALIDATION_DIAGNOSTICS_DIR> --diagnostics
.venv/bin/python scripts/train_classifier.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --resume-from-checkpoint <EXACT_CHECKPOINT_PATH>
```

Compatibility checks must match experiment, model revision, category-ID target,
prompt, taxonomy/stable-ID map, rank/targets, and subset lineage. Never select a
checkpoint using test data.

## L–N. Capacity decision and validation-only selection

Rank 8 is primary. Predeclare engineering acceptance criteria in a copy of
`configs/artifacts/candidate_acceptance.example.json`. Run rank 16 only if rank-8
validation macro F1 clearly plateaus below that criterion, invalid rate remains
materially above its criterion, or learning evidence indicates under-capacity.
Do not schedule ranks 32/64 by default.

```bash
.venv/bin/python scripts/select_final_candidate.py \
  --experiment-dir <RANK8_EXPERIMENT_DIR> \
  --output <RANK8_EXPERIMENT_DIR>/validation_selection.json
```

Selection order is higher validation macro F1, lower invalid rate, higher exact
ID rate, lower validation loss, then earlier checkpoint. If rank 16 is justified,
run its locked config and apply the same selector before comparing validation
reports only.

## GPU reset backup and recovery

Before a reset, put a complete `artifact_manifest.json` in the experiment root:

```bash
.venv/bin/python scripts/package_gpu_artifacts.py \
  --experiment <EXPERIMENT_DIR> \
  --output /content/clauseforge_gpu_backup.tar.gz
```

The archive allowlists adapter/checkpoint files, experiment metadata, validation
reports, selections, resume state, and environment. It excludes datasets,
secrets, caches, and unrelated files and performs no upload. Restore validates
every member path and checksum before extraction, then checks experiment ID,
target/prompt versions, stable-ID checksum, rank, and target modules.

## O–T. Import, validate, lock, authorize, and evaluate once

```bash
.venv/bin/python scripts/import_model_artifact.py \
  --adapter <SELECTED_ADAPTER_FILE_OR_ARCHIVE> \
  --manifest <ARTIFACT_MANIFEST>
.venv/bin/python scripts/validate_model_artifact.py --manifest <ARTIFACT_MANIFEST>
.venv/bin/python scripts/lock_final_model.py \
  --artifact-manifest <ARTIFACT_MANIFEST> \
  --validation-report <VALIDATION_SELECTION_JSON> \
  --output <FINAL_LOCK_JSON>
.venv/bin/python scripts/authorize_held_out_test.py \
  --lock <FINAL_LOCK_JSON> --artifact-manifest <ARTIFACT_MANIFEST> \
  --authorize-held-out-test
.venv/bin/python scripts/run_final_validation.py \
  --artifact-manifest <ARTIFACT_MANIFEST> --lock <FINAL_LOCK_JSON> \
  --authorize-held-out-test
```

That invocation prints the validated plan. For execution, create three reviewed
JSON files, each containing an argument array (never a shell string), for the
one-time model test evaluator, final safety command, and final EDGAR OOD command.
Each command must write its named JSON report. Then run:

```json
["/content/ClauseForge/.venv/bin/python", "<AUTHORIZED_TEST_EVALUATOR>", "--checkpoint", "<SELECTED_CHECKPOINT>", "--output", "<TEST_REPORT_JSON>"]
["/content/ClauseForge/.venv/bin/python", "scripts/run_safety_eval.py", "--provider", "configured", "--output", "<SAFETY_OUTPUT_DIR>"]
["/content/ClauseForge/.venv/bin/python", "scripts/run_edgar_ood.py", "--input", "<PREPARED_EDGAR_DIR>/segments.jsonl", "--output", "<OOD_OUTPUT_DIR>"]
```

Use `<SAFETY_OUTPUT_DIR>/summary.json` and `<OOD_OUTPUT_DIR>/summary.json` as the
corresponding reports. The authorized test evaluator must be the reviewed
model-checkpoint evaluator for the locked commit; it is intentionally not a
validation command or a classical baseline command.

Then run:

```bash
.venv/bin/python scripts/run_final_validation.py \
  --artifact-manifest <ARTIFACT_MANIFEST> --lock <FINAL_LOCK_JSON> \
  --authorize-held-out-test --execute \
  --test-command-json <TEST_COMMAND_JSON> --test-report <TEST_REPORT_JSON> \
  --safety-command-json <SAFETY_COMMAND_JSON> --safety-report <SAFETY_REPORT_JSON> \
  --ood-command-json <OOD_COMMAND_JSON> --ood-report <OOD_REPORT_JSON> \
  --environment-report <ENVIRONMENT_JSON> --output-bundle <FINAL_BUNDLE_JSON>
```

The command runs without a shell, stops on the first failure, consumes test
authorization only after the test command succeeds and its JSON is readable,
and refuses a repeat through the lock state.

The sequence is **select → import → validate → lock → authorize → execute
once**. Authorization is an explicit CLI flag bound to the locked identity.
Standard workflow refuses repeated authorization. The final-validation
orchestrator order is artifact validation, one authorized test, final safety,
final EDGAR OOD, then bundle generation. Preserve resulting reports and update
manifest states only from actual evidence.

## U–AC. Handoff, real execution, and release

After test/safety/OOD and the final evaluation bundle exist, plan the merge:

```bash
.venv/bin/python scripts/merge_adapter.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter <SELECTED_ADAPTER_DIR> --output <MERGED_MODEL_DIR> \
  --build-commit <LOCKED_GIT_COMMIT>
.venv/bin/python scripts/quantize_model.py --config <REVIEWED_QUANTIZATION_CONFIG>
.venv/bin/python scripts/validate_model_artifact.py --manifest <QUANTIZED_MANIFEST>
```

Existing merge/quantize commands emit validated plans; execute the reviewed
runtime-specific conversion only on appropriate compute and preserve
adapter→merged→quantized parent checksums.

Start the real configured service, then benchmark it:

```bash
.venv/bin/python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --requests <COUNT> --concurrency <COUNT> \
  --provider <PROVIDER> --model <MODEL> --backend <BACKEND> \
  --artifact-id <ARTIFACT_ID> --hardware <HARDWARE> \
  --quantization <QUANTIZATION>
```

The report captures request/concurrency, p50/p95/p99, throughput, failure rate,
and strict taxonomy-invalid rate. Do not use `--development` for a real model.

Container smoke tests are separate:

```bash
# CPU mock infrastructure smoke only
docker compose up --build
curl --fail http://127.0.0.1:8000/ready

# Real model smoke: provide the validated external manifest/artifact mounts
MODEL_ARTIFACT_MANIFEST=<CONTAINER_MANIFEST_PATH> docker compose -f <REVIEWED_REAL_COMPOSE_FILE> up --build
curl --fail http://127.0.0.1:8000/ready
```

Build the documented deployment bundle, validate its manifest/checksums, then
record deployment authorization as a distinct operator action only after every
gate passes:

```bash
.venv/bin/python scripts/authorize_deployment.py \
  --artifact-manifest <DEPLOYMENT_BUNDLE_MANIFEST> --authorize-deployment
```

Populate
`configs/artifacts/final_release_manifest.template.json` only with measured
evidence. Finish with:

```bash
.venv/bin/python scripts/release_status.py \
  --manifest <FINAL_ARTIFACT_MANIFEST> --lock <FINAL_LOCK_JSON>
```

Release is forbidden unless every status gate is complete.

## Failure recovery (fail closed)

- Colab reset/interruption: restore the verified backup, identify the exact
  numeric checkpoint, validate lineage, and pass that path explicitly.
- Corrupt checkpoint/archive/checksum: quarantine it; restore another verified
  copy. Never bypass validation.
- Wrong manifest/incompatible adapter: correct provenance or create a new
  candidate identity; never edit expected checksums to fit files.
- Partial quantization: discard the incomplete output and rerun from the
  validated parent; do not register it.
- Benchmark/container failure: retain failure evidence, keep gates blocked, fix
  the concrete defect, and rerun that gate.
# Pre-final candidate serving

Checkpoint 700 may be imported, validated, registered, activated, and served as
a real trained candidate before final locking. This does not authorize held-out
test evaluation, final safety/OOD, merge, quantization, final benchmarking, or
deployment. Checkpoints 1400 and 2100 follow the identical validation-only
comparison and artifact-activation path.
