# System overview

Serving flow: contract clause → strict input validation → configured provider →
classifier/model → exact 41-category taxonomy validation → optional exact cache
→ versioned response → privacy-preserving logs and process-local metrics.
Unknown output is rejected; there is no fuzzy mapping or silent mock fallback.

Training flow: CUAD → deterministic processing → contract-level train/validation/
test isolation → classical baseline → NF4 QLoRA → adapter → validation diagnostics
→ rank selection → one-time held-out test → merge → quantization → artifact
validation → deployment.

Implemented interfaces extend through deployment preparation. Unfinished steps
are checkpoint-25 diagnosis, full/rank training, model lock, held-out test,
final-model safety/OOD runs, merge, real quantization, real backend benchmarks,
and deployment.

