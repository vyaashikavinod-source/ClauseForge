# Quantization preparation

ClauseForge provides offline-safe planning and validation for two explicit
deployment paths. It does not bundle weights or perform quantization during
installation, tests, or service startup.

- AWQ: 4-bit weights, group size 128, declared calibration data, and a fixed
  calibration seed. `autoawq` is an optional external runtime dependency.
- GGUF: llama.cpp conversion followed by `Q4_K_M`, `Q5_K_M`, or `Q8_0`.
  The llama.cpp checkout and executables must be supplied explicitly.

Validate and print plans without loading a model:

```bash
python scripts/quantize_model.py --config quantization/configs/qwen25_7b_awq.yaml
python scripts/quantize_model.py --config quantization/configs/qwen25_7b_gguf_q4km.yaml
```

Adapter merging is separate. `merge_adapter.py` rejects base model, revision,
architecture, taxonomy, prompt, experiment, rank, or target-module mismatches.
Generated merged models, weights, and GGUF files belong under ignored
`artifacts/` paths.

Every artifact must carry immutable base/adapter revisions, checksums,
experiment ID, build commit, taxonomy/prompt versions, quantization parameters,
context length, validation state, and limitations. The deployment validator
checks the manifest and every required file checksum. This is not a model-quality
claim.

Heavy GPU/runtime packages are deliberately excluded from the base install.
Provision AutoAWQ, Transformers/PEFT, vLLM, or llama.cpp in a dedicated pinned
build image compatible with its CUDA and operating-system environment.

