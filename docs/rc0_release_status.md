# Deterministic RC0 release-status snapshot

Source: `configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json` with no lock.

```text
candidate: clauseforge-qwen25-7b-r8-rc0
lifecycle: release_candidate
artifact_valid: true
final_model_locked: BLOCKED
validation_complete: BLOCKED
held_out_test: BLOCKED
final_safety: BLOCKED
edgar_ood: BLOCKED
merge: BLOCKED
quantization: BLOCKED
benchmark: BLOCKED
container_smoke: BLOCKED
deployment_bundle: BLOCKED
deployment_authorized: BLOCKED
release_checklist: BLOCKED
release_allowed: false
```

Generate the full reasons and exact next actions with:

```bash
python scripts/release_status.py
python scripts/release_status.py --json
```
