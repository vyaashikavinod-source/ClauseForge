# Production serving backends

Backend selection is explicit through `CLAUSEFORGE_MODEL_BACKEND` (the legacy
`CLAUSEFORGE_MODEL_PROVIDER` remains compatible). There is no automatic
fallback to the mock provider.

| Backend | Intended artifact | Required configuration |
|---|---|---|
| `mock` | none | development only |
| `transformer` | base model plus PEFT adapter | model, adapter, tokenizer paths |
| `vllm` | merged/AWQ model served externally | vLLM URL and model identity |
| `llamacpp` | GGUF served externally | GGUF path and llama.cpp URL |

The vLLM and llama.cpp providers require their external server to pass
readiness and return one exact packaged taxonomy label. Unknown or malformed
output becomes a controlled provider error. Neither invents confidence scores.
API errors omit raw output, clause text, filesystem details, and upstream bodies.

Deployment manifests bind model/adapter lineage, quantization artifact, backend,
taxonomy/prompt versions, context limits, environment, checksums, and known
limitations. Validate an artifact before service startup:

```bash
python scripts/validate_deployment_artifact.py --manifest artifacts/deployment.json
```

Exact cache keys include normalized input plus provider, model revision, adapter,
taxonomy, prompt, and inference settings. The in-memory LRU reports hits,
misses, entries, evictions, and hit rate. `SemanticCache` is only a future typed
boundary; approximate reuse is disabled pending a legal-quality validation policy.

