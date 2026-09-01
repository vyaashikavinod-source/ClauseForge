"""Runtime evidence recorded beside each experiment."""

from __future__ import annotations

import platform
import sys

import peft
import torch
import transformers


def environment_metadata() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(0) if cuda else None),
    }
