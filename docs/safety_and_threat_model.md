# Safety and threat model

This describes implemented engineering controls, not proof of legal correctness
or production safety. Final trained-model safety performance remains pending
Phase 3B GPU execution.

## Trust boundaries

- **Clause text is untrusted data.** Structured prompt messages place it only in
  the user role. Instruction-like text receives no system privilege.
- **Provider output is untrusted.** Success requires exact membership in the
  versioned 41-label CUAD taxonomy. Surrounding whitespace is the only
  permissive normalization; aliases and fuzzy matches are rejected.
- **Providers are replaceable dependencies.** HTTP and evaluation depend on the
  provider protocol, not model-framework internals.
- **Logs are a privacy boundary.** They contain request IDs, route, provider,
  status, latency, size, and error class—not clause text or raw model output.
- **Legal scope is invariant.** Successful API output adds the centralized
  decision-support disclaimer. Requests and providers cannot suppress it.

## Threats covered

Synthetic fixtures exercise instruction-like content, "ignore previous
instructions" wording, label injection, buried signals, contradictions,
conflicting headings, whitespace and Unicode variation, repetition, degenerate
input, and cross-category ambiguity. Deterministic paraphrase pairs measure
consistency without asserting legal correctness.

The API controls unavailable providers, timeouts, unknown outputs, provider
exceptions, and input limits. Errors omit traces, paths, environment values,
and raw outputs.

## Not yet mitigated or proven

- Mock and keyword providers do not predict legal correctness.
- Role separation cannot prove a real model's injection resistance; the trained
  model must be evaluated with these and broader red-team cases.
- The small synthetic set does not represent all drafting styles, languages,
  jurisdictions, or adversaries.
- Authentication, rate limiting, tenant isolation, malware scanning, deployment
  hardening, and a raw HTTP-body byte limit are not implemented.
- No risk-advice, explanation-grounding, DPO, quantization, or production
  monitoring claim is made.

## Metrics

The harness reports adversarial pass rate, taxonomy-valid and invalid-output
rates, malformed-input rejection, provider safety failures, injection passes,
and paraphrase agreement. These are development harness measurements, not final
model-performance or legal safety metrics.
