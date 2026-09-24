from __future__ import annotations

import os
import subprocess
import sys


def test_mock_application_import_does_not_load_transformer_stack() -> None:
    """Keep the minimal deployment image independent of heavyweight ML packages."""
    environment = os.environ.copy()
    environment.update(
        {
            "CLAUSEFORGE_ENVIRONMENT": "production",
            "CLAUSEFORGE_MODEL_BACKEND": "mock",
            "CLAUSEFORGE_DEVICE": "cpu",
            "CLAUSEFORGE_METRICS_ENABLED": "false",
        }
    )
    environment.pop("MODEL_ARTIFACT_MANIFEST", None)
    command = (
        "import sys; "
        "from clauseforge.serving.app import app; "
        "assert app.state.provider.name == 'mock-development'; "
        "assert not any(name == 'torch' or name.startswith('torch.') "
        "for name in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
