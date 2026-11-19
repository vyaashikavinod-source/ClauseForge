"""Execute a reviewed local HF-to-GGUF conversion and Q4_K_M quantization.

Requires an already merged model and an existing official llama.cpp checkout.
Does not download anything or claim a successful export before verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged-model', type=Path, required=True)
    p.add_argument('--llama-cpp', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    merged = args.merged_model.resolve()
    checkout = args.llama_cpp.resolve()
    output = args.output.resolve()
    converter = checkout / 'convert_hf_to_gguf.py'
    candidates = [checkout / 'build/bin/llama-quantize', checkout / 'build/bin/Release/llama-quantize.exe', checkout / 'build/bin/llama-quantize.exe']
    quantizer = next((path for path in candidates if path.is_file()), None)
    if not converter.is_file() or quantizer is None or not (merged / 'config.json').is_file():
        raise ValueError('Need a merged model and built official llama.cpp converter/quantizer')
    if output.exists():
        raise ValueError('Refusing to overwrite an export')
    output.mkdir(parents=True)
    f16, q4 = output / 'model-f16.gguf', output / 'model-q4_k_m.gguf'
    start = time.perf_counter()
    subprocess.run([sys.executable, str(converter), str(merged), '--outfile', str(f16), '--outtype', 'f16'], check=True)
    subprocess.run([str(quantizer), str(f16), str(q4), 'Q4_K_M'], check=True)
    with q4.open('rb') as source:
        if source.read(4) != b'GGUF': raise ValueError('Export does not have a GGUF header')
    with q4.open('rb') as source:
        checksum = hashlib.file_digest(source, 'sha256').hexdigest()
    manifest = {'schema_version': 1, 'status': 'exported_not_evaluated', 'quantization': 'Q4_K_M',
                'file': q4.name, 'size_bytes': q4.stat().st_size, 'sha256': checksum,
                'runtime_seconds': time.perf_counter() - start,
                'llama_cpp_commit': subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
                'note': 'Export is not a quality or serving benchmark. Evaluate with the frozen harness.'}
    (output / 'export.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
