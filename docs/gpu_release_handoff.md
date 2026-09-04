# Deterministic checkpoint-800 release handoff

Source preparation is not a GPU result. No test, safety/OOD, merge, quantization,
benchmark, image build, or deployment has been certified by this work.

## Recovery rules

1. Restore the exact checkpoint-800 experiment and metadata from the verified
   private backup, using `scripts/restore_gpu_artifacts.py --archive ARCHIVE
   --output EMPTY_DESTINATION --expected-experiment-id
   qwen2.5-7b-instruct_lora-r8_seed42_0ad1c2c19e59`.
   This command fails on a nonempty destination. If the experiment already exists,
   do not restore over it; the preparer verifies its checkpoint/manifest checksums.
2. Recover `experiment_config.json`, checkpoint metadata/resume state, adapter
   metadata and original validation evidence from that backup. Reconstruction
   means restoring historical bytes, NOT guessing from current defaults.
   Missing historical metadata is a blocker.
3. Recover the original lock, manifest and attempt receipt as one unit from the
   durable runtime backup. Preserve absolute adapter paths. Once a lock has ever
   existed, do not initialize a replacement or move it without its receipt.
   The previously authorized checkpoint-800 runtime must restore its lock.
   For a snapshot produced by the new evidence backup tool, run:

   ```bash
   python scripts/restore_release_evidence.py --archive "$EVIDENCE_ARCHIVE" --output "$RESTORED_EVIDENCE"
   ```

   It verifies all bytes before writing, reuses identical files, and refuses to
   replace different existing state. Use `$RESTORED_EVIDENCE/artifact_manifest.json`
   and `$RESTORED_EVIDENCE/final_lock.json` (the attempt receipt is restored beside
   that lock). Restore checkpoint weights at their original absolute location.
4. If validation evidence is missing, restore it. Only for a genuinely fresh,
   never-locked selection may validation be rerun with the existing validation-only
   evaluator. Never recreate validation evidence to match an existing lock checksum.
   Validation is GPU work, and its report must not overwrite old evidence:

```bash
python scripts/evaluate_phase3b_checkpoint.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml \
  --data "$DATA" --checkpoint "$CHECKPOINT" \
  --output "$NEW_VALIDATION_DIRECTORY" --diagnostics
```

## One handoff preparation command

Run from the current source checkout on the GPU host. Set absolute paths to the
existing artifacts. `OUTPUT` must live on durable private storage if available.
Missing paths fail before inference. Do not run a second API/model process on the
same T4 while evaluating: free its VRAM using the existing server shutdown workflow.

```bash
export CHECKPOINT=/actual/restored/experiment/checkpoint-800
export MANIFEST=/actual/selected-artifact-manifest.json
export REGISTRY=/actual/artifacts/registry
export LOCK=/actual/existing-authorized-lock.json
export VALIDATION=/actual/original-validation-diagnostics.json
export DATA=/actual/data/processed/cuad/1.0.0-run-a
export EDGAR=/actual/existing-edgar-segments.jsonl
export OUTPUT=/actual/private/release-evidence

python scripts/prepare_gpu_release.py \
  --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --registry "$REGISTRY" \
  --lock "$LOCK" --validation-report "$VALIDATION" --data "$DATA" \
  --edgar "$EDGAR" --output "$OUTPUT"
```

This reuses and validates existing candidate/registry/lock state. For an entirely
new, never-locked release only, add BOTH:

```text
--initialize-selection "Intentionally selected checkpoint 800 using validation only before full configured training completion"
--authorize-held-out-test
```

These flags are NOT recovery options for a missing historical lock. If a candidate
manifest is absent during genuine fresh initialization, the preparer uses the
existing generic candidate creation and evidence attachment APIs. It never edits
runtime manifests to falsify training completion. The original training commit is
`a96fbab8c4f96a69bc40ad64ae1eee1354bab08e`.

Preparation generates deterministic command JSON files and an environment report.
It leaves valid existing files unchanged, rejects conflicting files, reuses an
existing authorization, and refuses an unconsumed attempt receipt. For completed
test state it emits `--resume-after-test`: that mode never invokes the test command.
It can refresh the active pointer's checksum for the same locked artifact after
in-place gate updates; it does not switch candidates or reset rollback history.

The printed `next_argv` is the exact command to execute. To prepare and execute
without shell interpretation, after confirming GPU availability:

```bash
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name(0))'
python - <<'PY'
import os, subprocess
from pathlib import Path
from clauseforge.release.handoff import prepare
names = ("CHECKPOINT", "MANIFEST", "REGISTRY", "LOCK", "VALIDATION", "DATA", "EDGAR", "OUTPUT")
plan = prepare(*(Path(os.environ[name]) for name in names))
if not plan["bundle_complete"]:
    subprocess.run(plan["next_argv"], check=True, shell=False)
PY
```

The test writes evidence only; the orchestrator validates it before consuming
authorization. Safety/OOD checks then use the same locked real transformer.
Exact category IDs are translated back to canonical taxonomy for these existing
harnesses; malformed IDs are never accepted. Safety requires zero adversarial
failures and zero paraphrase disagreements. OOD requires nonempty input and no
processing failures, but makes no accuracy claim on unlabeled data. A failing
safety report remains evidence of failure, not a completed release gate.

If safety/OOD fails or the session ends after test consumption, run preparation
again to obtain the resume command. It skips the held-out test and reuses successful
matching check reports. A failed existing report is not overwritten or silently
rerun: inspect the failure before further release work. An interruption between
the two one-time state writes remains a manual recovery blocker.

## Preserve evidence even when a later phase fails

```bash
python scripts/preserve_release_evidence.py \
  --artifact-manifest "$MANIFEST" --lock "$LOCK" --evidence "$OUTPUT" \
  --output /actual/durable/private/checkpoint800-release-SNAPSHOT.tar.gz
```

Use a new snapshot name each time. The checksum-indexed allowlist includes the
lock/attempt receipt, manifest, generated commands, test predictions, safety/OOD,
environment and bundle. It excludes weights, raw data, arbitrary files and secrets.
Keep the original experiment/weights backup and original validation evidence too;
this evidence archive is not a replacement for them. Verify archive checksums
before restoring, and never overwrite a newer lock or receipt with an older copy.

## Remaining artifact and deployment plans

After final checks pass, inspect without changing release flags:

```bash
python scripts/release_status.py --manifest "$MANIFEST" --lock "$LOCK" --json
MODEL_ARTIFACT_MANIFEST="$MANIFEST" python scripts/check_release_readiness.py --lock "$LOCK" --strict --json
python scripts/model_artifact_registry.py --registry "$REGISTRY" promote \
  clauseforge-qwen25-r8-step800-recovery-v1 final_candidate --lock "$LOCK"
```

Promotion requires the registry manifest's actual gate state; if evaluation used
a separate manifest file, stop and reconcile using the validated registry workflow,
not by copying boolean flags. Prefer the registered manifest from the start.

Existing merge and quantization **planning** commands:

```bash
python scripts/merge_adapter.py --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter "$CHECKPOINT" --output "$MERGED_OUTPUT" --build-commit "$SOURCE_COMMIT"
python scripts/quantize_model.py --config quantization/configs/qwen25_7b_awq.yaml
python scripts/quantize_model.py --config quantization/configs/qwen25_7b_gguf_q4km.yaml
```

These are not conversion executors. Merge planning requires the original
`adapter_metadata.json`; do not synthesize it from an incompatible schema. Review
the existing quantization configs for actual restored paths before execution.
AWQ still needs a conversion executor and non-test calibration data. GGUF emits
llama.cpp conversion commands but needs a pinned installed llama.cpp toolchain and
a genuine merged model. Full FP16 7B merge can exceed available T4 memory; do not
assume it fits. These are explicit remaining engineering/runtime blockers.

Deployment bundle planning uses the existing
`clauseforge.artifacts.workflows.build_deployment_bundle(output, model_manifest,
deployment_manifest, documentation)` API. It requires matching model identity and
checksums; validate with `scripts/validate_deployment_artifact.py --manifest PATH`.
Do not construct a deployment manifest until real merged/quantized artifact
checksums exist. No default deployment target is authorized.

## Isolated real container smoke (not a deployment)

`Dockerfile.gpu` and `docker-compose.release.yml` are source-prepared, NOT built or
GPU-verified. They keep the default mock Compose unchanged, require GPU support,
require an explicit proven bitsandbytes version, mount artifacts read-only at
their original absolute paths, and bind only loopback. Ensure the nonroot user
can read the mounted artifacts. This profile serves the real adapter; it is not
evidence for a future AWQ/GGUF production backend.

```bash
export MODEL_ARTIFACT_MANIFEST="$MANIFEST"
export RELEASE_ROOT=/actual/restored/artifact/root
export BNB_VERSION=VERSION_FROM_THE_PROVEN_GPU_ENVIRONMENT
docker compose -f docker-compose.release.yml config
docker compose -f docker-compose.release.yml up --build -d
# Wait for model readiness, then use a fresh isolated rate-limit window.
python scripts/smoke_release_http.py --url http://127.0.0.1:8000 \
  --artifact-id clauseforge-qwen25-r8-step800-recovery-v1 \
  --checkpoint-step 800 --output "$OUTPUT/container-smoke.json"
docker compose -f docker-compose.release.yml logs --no-color api
docker compose -f docker-compose.release.yml down
```

The smoke script verifies health/readiness/frontend/docs/metrics, real identity,
request/security headers, exact cache reuse and rate limiting. Inspect structured
request logs separately; the script truthfully records `logs_verified=false`.
Do not mark container smoke complete until build and log evidence also pass.
For the real benchmark use the existing `scripts/benchmark_serving.py --help`
contract, a suitable rate-limit policy, and the actual final backend/artifact;
operator-supplied labels alone are not proof of artifact identity.

Complete `docs/release_checklist.md` only from actual evidence. Quantization,
benchmark, container, deployment bundle and authorized deployment remain blocked
until their corresponding real operations succeed. No released status is granted
by this handoff script.
