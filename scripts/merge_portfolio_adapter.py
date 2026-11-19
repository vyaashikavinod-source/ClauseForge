"""Merge an actual local QLoRA adapter into a pinned public base model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--base-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--revision', default='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not (args.adapter / 'adapter_config.json').is_file(): raise ValueError('Actual adapter weights/config required')
    if args.output.exists(): raise ValueError('Refusing to overwrite merged weights')
    config = json.loads((args.adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path') != args.base_model: raise ValueError('Adapter/base identity mismatch')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision): raise ValueError('Pin an immutable model revision')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload(safe_merge=True)
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, trust_remote_code=False).save_pretrained(args.output)
    (args.output / 'merge-evidence.json').write_text(json.dumps({'status': 'merged_not_evaluated', 'base_model': args.base_model, 'base_revision': args.revision, 'duration_seconds': time.perf_counter() - started}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
