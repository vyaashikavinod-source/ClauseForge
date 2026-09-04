# Locked one-time held-out evaluation

This is SOURCE ENGINEERING until executed on the restored, authorized GPU
runtime. No local test results in this change represent CUAD model performance.

`scripts/evaluate_locked_held_out_test.py` is a release-only boundary. It validates
the lock, exact original manifest checksum, selection fingerprint, checkpoint
step, adapter checksum/files, pinned Qwen revision, prompt/target/taxonomy identity,
and unconsumed authorization before opening any dataset. It uses the existing
`LocalTransformerProvider`, CUDA, NF4/double quantization/FP16 and exact category-ID
validation. It has no mock, training, split selection, or checkpoint selection flag.
Legacy locks without a selection fingerprint and checkpoint/adapter identity are
rejected; do not recreate an authorized lock to work around this check.

Only test contracts are retained from the existing shared clauses file. Train and
validation records are not inferred or scored. The loader rejects overlapping
contract splits and unknown targets. No training or comparison path imports this
loader. The dataset layout is unchanged.

Default generation is deterministic (`do_sample=false`), sequence length 1024,
24 new tokens, matching the corrected category-ID v2 validation configuration.
Both limits are recorded. Do not tune them against test results.

## State and evidence

The evaluator never changes the lock or manifest. Before opening test data it
exclusively creates `<lock filename>.held-out-attempt.json` next to the lock.
This durable attempt receipt prevents concurrent/repeated inference even before
the orchestrator consumes authorization. Output/prediction files are also
non-overwriting. Keep the receipt, including after failure/interruption. There is
no retry/force option: do not delete it or copy the lock to bypass the guard.
If a run fails, retain all evidence and stop for investigation; false
`test_evaluated` is not permission to repeat a partially executed test.

The report schema is `clauseforge-held-out-final-test-v1`, label `HELD-OUT FINAL
TEST`, split `test`. It records artifact/checkpoint/experiment identity, adapter
and manifest checksums, selection fingerprint, pinned base/prompt/target/taxonomy
lineage, test-content and split-assignment checksums, timestamp, counts, category
coverage, accuracy, macro/weighted F1, exact-ID and invalid counts/rates,
malformed/empty counts, prediction distribution, generation configuration,
prediction-file checksum, and incomplete-training selection provenance.
`exact_id_count` counts valid IDs, not correct classifications; accuracy measures
correct classifications. Invalid predictions remain errors, never fuzzy-mapped.
Macro F1 uses the authoritative 41 IDs, including absent categories.

Raw predictions are separate JSONL records with clause ID, target ID, raw and
outer-whitespace-stripped output, status/reason, correctness, and `split=test`.
Loss and token-count diagnostics are null with an explicit limitation because
the production provider does not expose them. No estimates are invented.
Report `test_evaluated=true` is evidence of this completed inference run, not a
mutation of registry/lock state.

The final-validation orchestrator validates the successful command's parsed
report schema, lineage, counts and rates, then calls `record_test_evaluated` and
`mark_manifest_test_evaluated`. A failed command or invalid report leaves those
flags unchanged. These two existing writes are not a transaction: if interrupted
between them, stop and inspect; never reauthorize or rerun inference. Subsequent
safety/OOD failure likewise must not trigger another held-out run.

## GPU runtime sequence

First transfer the source commit (it is not pushed automatically). Restore the
existing checkpoint, manifest, authorized lock, processed CUAD and EDGAR artifacts
using the existing verified restore workflow. Do not recreate artifacts that
already exist. Set the following variables to their **actual restored paths**;
the local Windows checkout cannot determine the lock/evidence/EDGAR paths.

```bash
cd /content/ClauseForge
export MANIFEST=/content/ClauseForge/artifacts/registry/manifests/clauseforge-qwen25-r8-step800-recovery-v1.json
export LOCK=/absolute/path/to/existing-authorized-checkpoint-800-lock.json
export DATA=/content/ClauseForge/data/processed/cuad/1.0.0-run-a
export EDGAR=/absolute/path/to/existing-edgar-segments.jsonl
export FINAL=/absolute/path/to/new-final-evidence-directory

# Identity and GPU checks do not open CUAD.
python -c 'import os; from pathlib import Path; from clauseforge.evaluation.locked_test import verify_authorization; verify_authorization(Path(os.environ["LOCK"]), Path(os.environ["MANIFEST"])); print("Locked authorization verified")'
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name(0))'

export MODEL_ARTIFACT_MANIFEST="$MANIFEST"
export CLAUSEFORGE_MODEL_BACKEND=transformer
export CLAUSEFORGE_DEVICE=cuda
export CLAUSEFORGE_MAX_NEW_TOKENS=24

# Fail rather than overwrite an existing evidence directory.
# Validate paths and create exact JSON string arrays for the existing orchestrator.
python - <<'PY'
import json, os, platform, sys
from pathlib import Path
import torch
for name in ("MANIFEST", "LOCK", "EDGAR"):
    assert Path(os.environ[name]).is_file(), f"Missing {name}"
assert Path(os.environ["DATA"]).is_dir(), "Missing processed data directory"
root = Path(os.environ["FINAL"])
root.mkdir(parents=True, exist_ok=False)
commands = {
    "test-command.json": [sys.executable, "scripts/evaluate_locked_held_out_test.py",
        "--lock", os.environ["LOCK"], "--artifact-manifest", os.environ["MANIFEST"],
        "--data", os.environ["DATA"], "--output", str(root / "held-out-test.json"),
        "--predictions", str(root / "held-out-predictions.jsonl")],
    "safety-command.json": [sys.executable, "scripts/run_safety_eval.py",
        "--provider", "configured", "--output", str(root / "safety")],
    "ood-command.json": [sys.executable, "scripts/run_edgar_ood.py",
        "--input", os.environ["EDGAR"], "--output", str(root / "ood")],
}
for name, command in commands.items():
    (root / name).write_text(json.dumps(command, indent=2) + "\n")
(root / "environment.json").write_text(json.dumps({
    "python": platform.python_version(), "torch": torch.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
}, indent=2) + "\n")
PY

# Execute ONCE. The test authorization already exists; do not authorize again.
python scripts/run_final_validation.py \
  --artifact-manifest "$MANIFEST" --lock "$LOCK" \
  --authorize-held-out-test --execute \
  --test-command-json "$FINAL/test-command.json" \
  --test-report "$FINAL/held-out-test.json" \
  --safety-command-json "$FINAL/safety-command.json" \
  --safety-report "$FINAL/safety/summary.json" \
  --ood-command-json "$FINAL/ood-command.json" \
  --ood-report "$FINAL/ood/summary.json" \
  --environment-report "$FINAL/environment.json" \
  --output-bundle "$FINAL/final-evaluation-bundle.json"

python scripts/release_status.py --manifest "$MANIFEST" --lock "$LOCK" --json
```

The configured safety harness currently labels its report as development safety;
retain its original label and actual results. Successful execution is not proof
that every safety gate passed. Review results before promotion. This change does
not certify or redesign the existing safety/OOD harnesses.

Preserve the entire evidence directory, original selection evidence, updated
lock/manifest, and attempt receipt in the existing private artifact backup.
Never commit test predictions, datasets, weights, or runtime evidence to Git.
