# ClauseForge handoff

ClauseForge delivers reproducible data/baseline/training infrastructure,
GPU-feasibility evidence, checkpoint diagnostics, strict serving and safety
boundaries, OOD preparation, quantization plans, Docker/CI, observability, and
release operations. It does not deliver a finalized production model.

## Run and locate

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
uvicorn clauseforge.serving.app:app --host 127.0.0.1 --port 8000
python scripts/check_release_readiness.py --json
```

Configuration is in `.env.example`, typed settings, and `training/configs/`.
Generated adapters/checkpoints belong under ignored `checkpoints/`; merged and
quantized artifacts belong under ignored `artifacts/`. Never commit them.

Checkpoint-10 diagnostics mean the short pilot learned no exact-valid output at
that point; short-name recognition remains diagnostic and invalid. They do not
measure the held-out test or final model quality. Resume GPU work exactly through
[the GPU plan](gpu_resume_plan.md). A final release requires every applicable
item in [the release checklist](release_checklist.md), immutable manifests and
checksums, and explicit authorization.

Remaining GPU/model work: checkpoint-25 diagnosis, target-representation
decision, bounded pilot, adequate full training, rank selection, model lock,
one-time test, final safety/OOD evaluation, merge, quantization, validation, and
real serving benchmarks.

API demo: call `/health`, `/ready`, `/version`, then:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/version
curl -X POST http://127.0.0.1:8000/v1/classify \
  -H "Content-Type: application/json" -H "X-Request-ID: demo-1" \
  -d '{"text":"This synthetic sample is governed by the laws of North Harbor."}'
```

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/v1/classify",
    json={"text": "This synthetic sample is governed by North Harbor law."},
)
print(response.json())
```

Use only synthetic inputs for demonstrations. Error examples and codes are in
[`demo/README.md`](../demo/README.md).
