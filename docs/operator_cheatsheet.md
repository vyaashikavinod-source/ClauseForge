# ClauseForge operator cheat sheet

```bash
python scripts/release_status.py [--json]
python scripts/validate_model_artifact.py --manifest <MANIFEST>
python scripts/model_artifact_registry.py activate <ARTIFACT_ID>
python scripts/model_artifact_registry.py rollback
python scripts/model_artifact_registry.py active
python scripts/compare_model_candidates.py --candidate-a <CURRENT_MANIFEST> --candidate-b <NEW_MANIFEST>
python scripts/smoke_test_active_model.py
python scripts/train_classifier.py --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml --data data/processed/cuad/1.0.0-run-a
python scripts/train_classifier.py --config training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml --data data/processed/cuad/1.0.0-run-a --resume-from-checkpoint <CHECKPOINT>
python scripts/select_final_candidate.py --experiment-dir <DIR> --output <SELECTION_JSON>
python scripts/lock_final_model.py --artifact-manifest <MANIFEST> --validation-report <SELECTION_JSON> --output <LOCK_JSON>
python scripts/authorize_held_out_test.py --lock <LOCK_JSON> --artifact-manifest <MANIFEST> --authorize-held-out-test
python scripts/run_final_validation.py --artifact-manifest <MANIFEST> --lock <LOCK_JSON> --authorize-held-out-test
python scripts/merge_adapter.py --base-model <MODEL> --base-revision <REVISION> --adapter <ADAPTER> --output <OUTPUT> --build-commit <COMMIT>
python scripts/quantize_model.py --config <CONFIG>
python scripts/benchmark_serving.py --base-url <URL> --requests <N> --concurrency <N> --provider <PROVIDER> --model <MODEL> --backend <BACKEND> --artifact-id <ID> --hardware <GPU> --quantization <TYPE>
python scripts/authorize_deployment.py --artifact-manifest <BUNDLE_MANIFEST> --authorize-deployment
# Build/validate the deployment bundle per docs/final_gpu_execution_runbook.md.
```

Real candidate startup requires `MODEL_ARTIFACT_MANIFEST=<MANIFEST>`,
`CLAUSEFORGE_MODEL_BACKEND=real`, and `CLAUSEFORGE_DEVICE=cuda`. The registry
preserves current and previous candidates; activation is not final promotion.
