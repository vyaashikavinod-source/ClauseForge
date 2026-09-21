from __future__ import annotations

from pathlib import Path

from clauseforge.training.cli import main


def test_cli_dry_run(capsys: object, processed_cuad_dir: Path) -> None:
    assert (
        main(
            [
                "--config",
                "training/configs/smoke.yaml",
                "--data",
                str(processed_cuad_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "dry-run-valid" in output
    assert "test_split_access" in output
