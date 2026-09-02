# GPU resume plan

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
