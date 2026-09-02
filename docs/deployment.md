# Deployment guide

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

