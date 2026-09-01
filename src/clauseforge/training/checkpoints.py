"""Small adapter checkpoints and inspectable experiment metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peft import PeftModel


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_adapter(model: Any, experiment_dir: Path) -> Path:
    adapter_dir = experiment_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    return adapter_dir


def reload_adapter(base_model: Any, adapter_dir: Path) -> Any:
    return PeftModel.from_pretrained(base_model, adapter_dir)
