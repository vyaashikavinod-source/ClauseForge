"""Serve a real local adapter on loopback for the common evaluation harness."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        raise ValueError("Actual trained adapter required")
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ready", "provider": "real-local-adapter"}

    @app.post("/v1/chat/completions")
    def completion(body: dict):
        messages = body.get("messages", [])
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise HTTPException(
                400, "The benchmark sends one pre-rendered shared prompt"
            )
        prompt = messages[0].get("content", "")
        if not isinstance(prompt, str) or len(prompt) > 50000:
            raise HTTPException(400, "Invalid prompt")
        with lock, torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            count = inputs["input_ids"].shape[1]
            limit = min(int(body.get("max_tokens", 64)), 128)
            if count + limit > 4096:
                raise HTTPException(
                    400, "Prompt exceeds the training context; no silent truncation"
                )
            started = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output[0][count:]
            return {
                "model": str(args.adapter.name),
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": tokenizer.decode(
                                generated, skip_special_tokens=True
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": count,
                    "completion_tokens": len(generated),
                    "total_tokens": count + len(generated),
                },
                "measurement": {
                    "latency_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None,
                },
            }

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
