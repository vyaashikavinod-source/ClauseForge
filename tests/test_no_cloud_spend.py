import os
from pathlib import Path
import subprocess
import sys


def test_modal_job_refuses_to_load_without_explicit_spending_authorization():
    environment = dict(os.environ)
    environment.pop('PORTFOLIO_ALLOW_CLOUD_SPEND', None)
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'modal_portfolio.py'
    result = subprocess.run([sys.executable, str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Cloud spending is disabled' in result.stderr
