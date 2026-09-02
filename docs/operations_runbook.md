# Operations runbook

ClauseForge is infrastructure-ready for mock validation, not production-model
ready. Final adapter selection, test evaluation, quantization, and benchmarks
remain blocked.

## Start and stop

Validate configuration with `python scripts/check_release_readiness.py --json`.
Start locally with `uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port
8000`, or use Docker Compose. Stop with the process supervisor's graceful
termination signal; FastAPI closes the provider once during shutdown.

`/health` proves process liveness. `/ready` checks the configured provider and
returns 503 when artifacts or an upstream server are unavailable. `/version`
returns safe build, taxonomy, prompt, provider, and backend identity. `/metrics`
is disabled by default and, when enabled, exposes process-local counters only;
it is intended for internal networks.

## Incidents

- `provider_unavailable`: verify external service or artifact provisioning; do
  not switch silently to mock.
- `inference_timeout`: inspect upstream latency and capacity before changing the
  bounded timeout.
- `invalid_model_output`: retain the raw output only in an approved offline
  diagnostic artifact, never normal logs; exact taxonomy validation stays strict.
- `artifact_mismatch` or `configuration_error`: stop startup and revalidate
  manifests, revisions, checksums, prompt, and taxonomy lineage.
- Cache anomalies: exact cache keys bind model, adapter, prompt, taxonomy, and
  inference settings. Clear process-local cache through a safe restart.

Logs are JSON on stdout/stderr for collection by the deployment platform. They
contain request metadata, not clause text or raw model output. Restart by
draining traffic, terminating gracefully, and replacing the process. Roll back
to the prior immutable application image and model-manifest pair. Replace model
artifacts by staging a new immutable directory, validating checksums/manifests,
starting a new instance, checking readiness, then shifting traffic; never mutate
live files in place.

