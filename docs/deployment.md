# Deployment guide

## Manifest-driven model swap and rollback

Set `MODEL_ARTIFACT_MANIFEST` to one validated external artifact manifest.
Startup rejects invalid checksums, incompatible taxonomy/prompt metadata, and
unsafe paths; there is no fallback. A future final adapter needs configuration
and manifest changes, not a serving-code rewrite.

The local registry stores manifests under ignored `artifacts/registry`.
`activate` writes a checksum-bound relative JSON pointer and preserves the
previous identity; `rollback` validates and restores it. JSON is used instead
of symlinks for portability. This is local state, not distributed orchestration.

```bash
python scripts/model_artifact_registry.py list
python scripts/model_artifact_registry.py inspect ARTIFACT_ID
python scripts/model_artifact_registry.py validate /path/to/manifest.json
python scripts/model_artifact_registry.py activate ARTIFACT_ID
python scripts/model_artifact_registry.py rollback
```

A future bundle contains `model_manifest.json`, `deployment_manifest.json`,
`checksums.json`, `config/`, and `docs/`. Binaries remain external.
Release also requires a real artifact-bound benchmark and container smoke test.

## CPU mock validation

Run `docker compose up --build`. The default image contains application code and
Python dependencies only: no datasets, credentials, adapters, or model weights.
The mock backend validates transport and operations, not model performance.

## Future model backends

- Transformer: mount immutable model, tokenizer, and adapter directories and set
  the corresponding `CLAUSEFORGE_*_PATH` values.
- vLLM: provision a separately managed GPU service, set its base URL and model
  identity, and validate the deployment manifest.
- llama.cpp: mount a checksum-validated GGUF read-only and configure the external
  server URL.

GPU images, orchestration, final adapters, real AWQ/GGUF artifacts, and serving
benchmarks are pending. The standard image is deliberately CPU/slim and performs
no download or fallback. Keep CORS off unless a reviewed browser client needs it.

Build metadata is injected with `CLAUSEFORGE_BUILD_COMMIT` and
`CLAUSEFORGE_BUILD_TIMESTAMP`; normal source artifacts contain no generated
timestamp. Generate an optional SBOM during a release build with CycloneDX or
Syft, for example `syft clauseforge:TAG -o cyclonedx-json > release/sbom.json`.
Generated SBOMs are release artifacts, not committed source.

Future release layout:

```text
release/
  application/   # generated wheel and image references
  manifests/     # reviewed deployment and quantization manifests
  model/         # sensitive/large externally stored artifacts, never Git
  docs/          # source-controlled runbook and deployment guide
  checksums/     # generated release checksums
```
