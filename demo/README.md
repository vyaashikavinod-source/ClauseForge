# Synthetic development demo

All text in this directory is original synthetic wording. It contains no CUAD
clause text or client contract material. Expected labels describe fixture intent
only and are not measured performance evidence.

```bash
python scripts/run_demo.py
```

The command runs ClauseForge in-process with the deterministic mock provider and
prints API-schema results including request ID, taxonomy version, processing
metadata, and disclaimer. Mock predictions demonstrate transport and provider
boundaries only; they are not expected to match every fixture label.

For an HTTP demo, start `uvicorn clauseforge.serving.app:app --host 127.0.0.1
--port 8000`, then call `/health`, `/ready`, `/version`, and `/v1/classify` using
only these synthetic texts. See [the handoff guide](../docs/handoff.md).

Deterministic error demonstrations:

- blank JSON text → HTTP 422 `request_validation_error`;
- text over `CLAUSEFORGE_MAX_INPUT_CHARACTERS` → HTTP 413 `input_too_large`;
- explicitly unconfigured real provider → HTTP 503 `provider_unavailable`;
- stub provider returning a non-taxonomy value → HTTP 422 `invalid_model_output`.

