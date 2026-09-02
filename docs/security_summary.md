# Security and privacy summary

- Raw clauses, client documents, and normal raw model outputs are prohibited in
  logs; structured logs retain safe request metadata only.
- Caller request IDs are length/character validated or replaced, returned in
  headers/errors, and protected against log injection.
- Errors are stable and sanitized; internal exceptions and paths are not exposed.
- Secrets enter through deployment configuration and are excluded from images
  and Git. Data, weights, adapters, checkpoints, caches, and generated results
  are ignored.
- Providers are explicit and isolated; readiness fails instead of falling back.
- Strict schemas, input limits, timeouts, security headers, exact taxonomy
  checks, and prompt-injection robustness fixtures provide defense in depth.
- Deployment assumes TLS, network controls, secret management, immutable
  artifacts, and external distributed rate limiting where required.

This is an engineering summary, not a formal audit, certification, or guarantee.

