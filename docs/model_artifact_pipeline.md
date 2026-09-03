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

Use `configs/artifacts/clauseforge-qwen25-7b-r8-step700.template.json` as the
schema guide. Put the completed manifest beside the restored external
checkpoint, set its relative adapter path, and replace all checksum/provenance
placeholders. Validation checks safetensors, PEFT config, checkpoint metadata,
optional resume state, Qwen revision, LoRA structure, prompt/target versions,
stable-ID map, experiment identity, and checkpoint step—not the filename.

```bash
python scripts/import_model_artifact.py --adapter <RESTORED_CHECKPOINT_700> --manifest <MANIFEST_700>
python scripts/validate_model_artifact.py --manifest <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> validate <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> register <MANIFEST_700>
python scripts/model_artifact_registry.py --registry <REGISTRY> activate clauseforge-qwen25-7b-r8-step700
python scripts/model_artifact_registry.py --registry <REGISTRY> inspect clauseforge-qwen25-7b-r8-step700
python scripts/release_status.py --manifest <MANIFEST_700>
```

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
