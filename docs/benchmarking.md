# Serving benchmark methodology

`scripts/benchmark_serving.py` sends bounded concurrent requests to an already
running service. It records request count, concurrency, success/failure/timeout
and invalid counts, p50/p95/p99 and mean latency, throughput, runtime, provider,
model, backend, and environment metadata.

```bash
python scripts/benchmark_serving.py --base-url http://127.0.0.1:8000 \
  --requests 100 --concurrency 4 --timeout 30 --warmup 5 \
  --provider vllm --model Qwen/Qwen2.5-7B-Instruct --backend awq
```

Use a fixed manifest, hardware image, corpus, request count, concurrency,
timeout, and warmup count for comparisons. Report failures and timeouts rather
than discarding them. Latency is wall-clock client latency including transport.

Mock runs require `--development` and are labeled `DEVELOPMENT HARNESS — NOT
MODEL PERFORMANCE`. They validate load generation and metric calculations only.
The tool does not load a model, execute held-out CUAD evaluation, or claim
legal-quality performance.
