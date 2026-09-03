# GPU resume plan

## Compressed strategy for limited compute

The memorization diagnostic and corrected 100-step validation pilot completed.
Continue rank 8 first: run larger/full corrected v2 training, evaluate
validation, and lock rank 8 if it satisfies predeclared engineering criteria.
Try rank 16 only if rank 8 clearly plateaus or misses those criteria. Avoid a
broad rank sweep unless durable compute is available. This does not change
seeds, lineage, strict-ID evaluation, or the sealed-test policy.

Do not run another normal pilot yet. First run this GPU-only memorization test:

```bash
python scripts/train_classifier.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --overfit-diagnostic \
  --overfit-examples 8 \
  --overfit-steps 100
```

For a shorter infrastructure smoke use `--overfit-steps 10`. If the model cannot
memorize eight examples, treat the pipeline or optimization as incompatible and
do not run a normal pilot. If it can, investigate sample size, learning rate,
duration, and generalization next. Memorization is not model quality.
The equivalent dedicated overfit config is
`training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_overfit.yaml`.

The overfit run succeeded: all eight training IDs were exact by step 20 and
stayed exact through step 100. The next experiment is:

```bash
python scripts/train_classifier.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --pilot \
  --pilot-train-examples 256 \
  --pilot-validation-examples 128 \
  --max-steps 100
```

The config validates and checkpoints at steps 20, 40, 60, 80, and 100.

1. Obtain GPU access and preserve historical pilot artifacts.
2. Run the bounded category-ID v2 validation pilot above with a new adapter.
3. Inspect strict ID diagnostics and checkpoints; do not access test.
4. Select configuration using validation only, then obtain durable compute.
5. Complete rank selection, lock one model/configuration, and only then perform
   the one-time held-out test, safety, and SEC EDGAR OOD evaluations.
15. Merge the adapter.
16. Quantize.
17. Validate the artifact and lineage.
18. Benchmark the real serving backend.
19. Complete the release checklist.

Do not overwrite historical artifacts or reinterpret new validation samples as
the original pilot. The held-out test remains sealed through step 11.
