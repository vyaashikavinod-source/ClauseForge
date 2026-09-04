# Model artifact and release-candidate pipeline

ClauseForge uses `clauseforge-artifact-v1` JSON manifests for adapters, merged
models, AWQ/GGUF outputs, and deployment bundles. Manifests bind base model and
revision, experiment, adapter checksum/path, category-ID representation,
prompt/taxonomy/stable-ID versions, LoRA structure, training commit/config,
validation evidence, lifecycle status, required-file checksums, limitations,
and parent identity/checksum. Paths are relative and containment-checked.

## Offline workflow

```bash
python scripts/validate_model_artifact.py --manifest MANIFEST
python scripts/import_model_artifact.py --adapter EXTERNAL_FILE --manifest MANIFEST
python scripts/model_artifact_registry.py --registry artifacts/registry list
```

Import validates but never copies model data into source control. The registry
supports `list`, `inspect`, `validate`, `promote`, `activate`, and `rollback`.
Lifecycle transitions are `pilot -> release_candidate -> final_candidate ->
released`; transitions cannot skip states.

Final-candidate gates require completed full training, locked configuration,
completed validation selection, one authorized held-out test, final safety and
EDGAR OOD results, plus valid artifact lineage. Released gates additionally
require a validated merged/quantized deployment bundle, real serving benchmark,
container smoke, deployment manifest, checklist, and authorization. RC0 visibly
fails these gates.

`lock_final_model.py` freezes experiment/commit/config/taxonomy/prompt identity
before test phase. Changing any locked configuration requires a new candidate.
The one-time state transition from `test_evaluated=false` to `true` rejects a
second standard-workflow evaluation. It does not itself open or evaluate data.

`run_final_validation.py` only emits an auditable future plan: artifact
validation, explicitly authorized one-time test, final safety harness, final
EDGAR OOD, then a final evaluation bundle. The bundle schema keeps validation,
test, safety, OOD, environment, commit, artifact, timestamp, and limitations
separate. RC0 has no fabricated placeholders for unavailable results.

Quantization lineage is adapter -> merged model -> AWQ/GGUF -> deployment
bundle. Each derived manifest must bind its parent artifact ID and manifest
checksum. Benchmark reports accept artifact ID, backend, quantization, hardware,
request/concurrency, latency percentiles, throughput, failure rate, and strict
taxonomy-invalid rate; no numbers are generated without execution.

Candidate acceptance settings in `configs/artifacts/candidate_acceptance.example.json`
are configurable engineering gates, not production thresholds. Null criteria do
not block RC0 merely because arbitrary thresholds have not been invented.

## Hot-swappable real candidates

Use `scripts/create_model_candidate.py` for every compatible restored
checkpoint; no checkpoint-specific source or template is needed. Put the output
manifest at the operator-selected path. The command stores the resolved external
adapter path, derives its checksum, and cross-checks persisted lineage. Paths
are never exposed by readiness or classification responses. Validation checks safetensors,
PEFT config, checkpoint metadata,
optional resume state, Qwen revision, LoRA structure, prompt/target versions,
stable-ID map, experiment identity, and checkpoint step—not the filename.

```bash
python scripts/create_model_candidate.py --checkpoint <RESTORED_CHECKPOINT> --artifact-id <ARTIFACT_ID> --output <CANDIDATE_MANIFEST> --training-commit <HISTORICAL_TRAINING_COMMIT>
python scripts/attach_candidate_validation.py --manifest <CANDIDATE_MANIFEST> --validation-evidence <VALIDATION_JSON> --output <VALIDATED_MANIFEST>
python scripts/import_model_artifact.py --adapter <RESTORED_CHECKPOINT_700> --manifest <MANIFEST_700>
python scripts/validate_model_artifact.py --manifest <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> validate <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> register <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> activate clauseforge-qwen25-7b-r8-step700
python scripts/model_artifact_registry.py --registry <REGISTRY> inspect clauseforge-qwen25-7b-r8-step700
python scripts/release_status.py --manifest <MANIFEST_700>
```

Creation accepts checkpoint 700, 800, 1400, 2100, or any later compatible
checkpoint without code changes. If checkpoint metadata has no validation
summary, creation intentionally leaves `validation_summary` null. The separate
attachment command accepts validation split evidence only, requires the same
checkpoint step, and never accepts held-out test evidence. A trained candidate
without attached validation evidence cannot be compared or activated.

For 1400 or 2100, validate a new manifest, compare validation evidence,
register it, then activate only if accepted. The pointer retains current and
previous candidates, so rollback is immediate.

```bash
python scripts/compare_model_candidates.py --candidate-a <CURRENT_MANIFEST> --candidate-b <NEW_MANIFEST>
python scripts/model_artifact_registry.py --registry <REGISTRY> activate <NEW_ARTIFACT_ID>
python scripts/model_artifact_registry.py --registry <REGISTRY> rollback
python scripts/model_artifact_registry.py --registry <REGISTRY> active
```

Comparison never reads test data. Ranking is macro F1, lower invalid-output
rate, higher exact-ID rate, lower validation loss, then earlier checkpoint.
Activation does not promote or final-lock a candidate.

Real serving sets `MODEL_ARTIFACT_MANIFEST=<MANIFEST>`,
`CLAUSEFORGE_MODEL_BACKEND=real`, and `CLAUSEFORGE_DEVICE=cuda`. It loads the
pinned Qwen base in 4-bit NF4 with double quantization and FP16 compute, attaches
the PEFT adapter, uses the manifest prompt, and accepts only an exact category
ID. Failures never fall back to mock. `python scripts/smoke_test_active_model.py`
uses synthetic clauses and labels results **NOT FINAL MODEL PERFORMANCE**.
## Intentionally selected incomplete training

An operator may select a validated checkpoint before the configured training
trajectory ends. This does **not** set `full_training_completed=true`.
`lock_final_model.py --allow-incomplete-training-selection REASON` is explicit
release authorization, not an automatic inference from incomplete training.
It also requires `--selected-artifact-id` matching the manifest, a
`release_candidate`, completed validation selection, validation-only evidence
matching its metrics and lineage, no prior test evaluation, and successful
artifact/file/checksum validation. Use the active registry manifest, not a
newly reconstructed manifest. Preserve the evidence and lock with the model.

The lock records the operator reason, original training-completion flag,
checkpoint step, adapter checksum, original manifest checksum, validation-report
checksum, and a selection fingerprint covering identity/configuration and
validation evidence. Only subsequent release-result/status fields are excluded
from the fingerprint. Existing locks cannot be overwritten to reset test state.
Old fully-trained locks remain readable; normal fully-trained locking needs no
exception. No artifact-manifest schema or historical training field is changed.

For checkpoint 800 in the GPU runtime, set `VALIDATION_REPORT` to its existing
machine-readable validation evidence (do not regenerate it merely for locking):

```bash
python scripts/lock_final_model.py \
  --artifact-manifest /content/ClauseForge/artifacts/registry/manifests/clauseforge-qwen25-r8-step800-recovery-v1.json \
  --validation-report "$VALIDATION_REPORT" \
  --output /content/ClauseForge/artifacts/final/checkpoint-800-lock.json \
  --selected-artifact-id clauseforge-qwen25-r8-step800-recovery-v1 \
  --allow-incomplete-training-selection "Intentionally selected checkpoint 800 using validation-only evidence before completion of the configured three-epoch trajectory"
```

The lock does not authorize or execute the test. Use the existing explicit
one-time authorization workflow next. A locked `release_candidate` can enter
final validation without prematurely becoming `final_candidate`; otherwise
promotion's test/safety/OOD prerequisites would create a circular dependency.
Authorization/consumption remains one-time. The reason is retained in the
final-evaluation bundle and `release_status.py --lock ...` output.

For later promotion, pass `--lock` to `model_artifact_registry.py ... promote
ARTIFACT_ID final_candidate --lock LOCK_PATH`. The validated, identity-matching
lock satisfies only the training-completion prerequisite. Without it incomplete
training still blocks promotion. Test, safety, OOD, and every subsequent
quantization, benchmark, container, deployment, and release gate remain required.
The exception is bound to the selected artifact, not a blanket authorization
for other checkpoints or derived artifacts.
