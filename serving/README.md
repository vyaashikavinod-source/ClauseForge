# Serving

The serving/API foundation is complete. It is an inference-service boundary,
not a deployed product or evidence of trained-model performance.

## Architecture

- FastAPI owns versioned HTTP schemas and OpenAPI documentation.
- `ClauseClassifierProvider` separates HTTP behavior from model frameworks.
- `mock-development` is the explicit deterministic local default.
- `local-transformer` fails readiness until real artifacts are provisioned.
- The packaged `cuad-v1-41` taxonomy validates every successful result.
- Structured errors omit traces, paths, environment values, and raw outputs.
- Logs contain request IDs, route/status/latency, and safe size metadata only.

No permissive CORS policy is installed. Oversized input is rejected rather than
truncated.

## Run locally

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:CLAUSEFORGE_MODEL_PROVIDER = "mock"
uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
source .venv/bin/activate
CLAUSEFORGE_MODEL_PROVIDER=mock \
  uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port 8000
```

Routes are `GET /health`, `GET /ready`, `POST /v1/classify`, and `GET /docs`.
The mock provider exposes no fabricated confidence or scores.

Transformer mode requires `CLAUSEFORGE_MODEL_PATH`,
`CLAUSEFORGE_ADAPTER_PATH`, and `CLAUSEFORGE_TOKENIZER_PATH`. Missing artifacts
produce unavailable readiness. Actual transformer loading remains pending a
trained adapter and CUDA-capable host.

This system assists with contract clause analysis and does not provide legal
advice.
