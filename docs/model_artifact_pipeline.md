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
