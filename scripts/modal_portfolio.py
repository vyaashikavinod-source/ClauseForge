"""Budget-bounded Modal QLoRA jobs. No cloud work runs on import.

First run: modal run scripts/modal_portfolio.py --mode pilot
Training run: modal run scripts/modal_portfolio.py --mode train
Verify account credits/billing before invoking either remote entrypoint.
"""
from __future__ import annotations

from pathlib import Path
import os

if os.environ.get('PORTFOLIO_ALLOW_CLOUD_SPEND') != 'explicitly-authorized':
    raise RuntimeError('Cloud spending is disabled. Use the free Colab notebook; do not launch Modal jobs without fresh authorization.')

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/cuad/portfolio-v1'
CONFIG = ROOT / 'training/configs/portfolio_qwen15b_qlora.yaml'
app = modal.App('clauseforge-portfolio-evidence')
artifacts = modal.Volume.from_name('clauseforge-portfolio-artifacts', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.13.0', 'transformers==5.16.1', 'peft==0.20.0',
                 'bitsandbytes==0.49.0', 'accelerate', 'PyYAML==6.0.2',
                 'numpy==2.2.6', 'scikit-learn==1.9.0', 'fastapi==0.141.1')
    .env({'PYTHONPATH': '/app/src', 'HF_HOME': '/artifacts/huggingface'})
    .add_local_dir(ROOT / 'src', '/app/src', copy=True)
    .add_local_file(CONFIG, '/app/config.yaml', copy=True)
    .add_local_dir(DATA, '/data/cuad', copy=True)
)


def execute(mode: str):
    from dataclasses import replace
    import importlib.metadata
    import json
    import time
    import torch
    from clauseforge.training.cli import run
    from clauseforge.training.config import load_config

    if mode not in {'pilot', 'train'}:
        raise ValueError('Choose pilot or train')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    output = Path('/artifacts') / mode
    config = load_config(Path('/app/config.yaml'))
    config = replace(config, output_dir=output,
                     data=replace(config.data, max_validation_examples=64))
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = run(config, Path('/data/cuad'), dry_run=False,
                     pilot=mode == 'pilot', pilot_train_examples=64,
                     pilot_validation_examples=32, max_steps=25 if mode == 'pilot' else 500)
        summary = {'mode': mode, 'status': 'completed', 'test_evaluated': False,
                   'step_limit': 25 if mode == 'pilot' else 500,
                   'gpu': torch.cuda.get_device_name(0), 'duration_seconds': time.time() - started,
                   'packages': {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes')},
                   'training': result}
        (output / 'modal-run.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        return summary
    finally:
        artifacts.commit()


@app.function(image=image, gpu='T4', cpu=2, memory=12288, timeout=1800,
              retries=0, max_containers=1, volumes={'/artifacts': artifacts})
def pilot():
    return execute('pilot')


@app.function(image=image, gpu='T4', cpu=2, memory=12288, timeout=21600,
              retries=0, max_containers=1, volumes={'/artifacts': artifacts})
def train():
    return execute('train')


@app.local_entrypoint()
def main(mode: str = 'pilot'):
    import json
    if mode not in {'pilot', 'train'}:
        raise ValueError('Choose pilot or train')
    result = pilot.remote() if mode == 'pilot' else train.remote()
    print(json.dumps(result, indent=2, default=str))
