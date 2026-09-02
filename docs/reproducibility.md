# Reproducibility guide

Use Python 3.11, create a virtual environment, install `requirements.txt`, and
run `make check` (or the equivalent Python commands). Package imports and tests
are CPU/offline. The synthetic demo is `python scripts/run_demo.py`.

CUAD preparation requires an explicitly downloaded public source:

```bash
python scripts/prepare_cuad.py --input data/raw/cuad/CUAD_v1.json \
  --output data/processed/cuad/1.0.0
python scripts/run_baselines.py --help
```

Manifests capture source/split checksums. Splits use deterministic contract-level
assignment. Training and pilot sampling use seed 42 in Phase 3B configs. GPU
reproduction requires a compatible CUDA host, pinned Qwen revision, bitsandbytes,
NF4/double quantization, FP16, and the recorded LoRA configuration. Use explicit
training and checkpoint-evaluation CLIs only on that host; preserve experiment,
selection, resume, and adapter metadata.

Normal reproduction uses train and validation only. Do not run held-out test as
a routine check. Resume full Phase 3B and the one-time test only according to
[the GPU resume plan](gpu_resume_plan.md).

