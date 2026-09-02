# API error taxonomy

| Code | HTTP | Meaning |
|---|---:|---|
| `request_validation_error` | 422 | Strict schema or malformed JSON failure |
| `input_too_large` | 413 | Configured text limit exceeded |
| `rate_limit_exceeded` | 429 | Process-local development limiter rejected request |
| `provider_unavailable` | 503 | Configured provider is not ready |
| `inference_timeout` | 504 | Provider exceeded its deadline |
| `invalid_model_output` | 422 | Output was not an exact taxonomy value |
| `artifact_mismatch` | 409 | Artifact lineage/checksum mismatch during validation |
| `configuration_error` | 500 | Deployment configuration is invalid |
| `internal_error` | 500 | Sanitized unexpected failure |

Every API failure includes a safe request ID. Internal exception messages,
paths, clause content, and model output are not returned.

