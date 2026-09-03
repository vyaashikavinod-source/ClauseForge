# Release checklist

- [ ] Tests pass
- [ ] Ruff lint and format checks pass
- [ ] mypy strict passes
- [ ] Secret and tracked-artifact scans pass
- [ ] Artifact lineage and checksums are valid
- [ ] Final adapter imported
- [ ] Artifact validated
- [ ] Candidate locked
- [ ] Final adapter selected
- [ ] Rank sweep complete
- [ ] Held-out test evaluated exactly once under the approved protocol
- [ ] Safety evaluation rerun on the final model
- [ ] SEC OOD evaluation run on the final model
- [ ] Quantization executed
- [ ] Quantized artifact validated
- [ ] Serving benchmark executed
- [ ] Container smoke test passes
- [ ] Deployment manifest validated
- [ ] Release bundle built
- [ ] Deployment authorized
- [x] Legal-assistance disclaimer present

Unchecked items are current blockers. This checklist must not be auto-completed
from infrastructure-only or mock-backend evidence.

Use `python scripts/release_status.py` for reasons and exact next actions. The
operator records deployment authorization separately with
`scripts/authorize_deployment.py --authorize-deployment`; it fails closed until
all preceding gates are evidenced.
