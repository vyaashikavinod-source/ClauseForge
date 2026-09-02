# GPU resume plan

Next validation-only pilot (GPU only; do not run locally):

```bash
python scripts/train_classifier.py \
  --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml \
  --data data/processed/cuad/1.0.0-run-a \
  --pilot --pilot-train-examples 256 \
  --pilot-validation-examples 128 --max-steps 25
```

Engineering success means exact valid IDs above zero, invalid rate below 1.0,
visible category structure, decreasing loss, checkpoints, and zero test access.
These are pilot criteria, not final quality thresholds.

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
