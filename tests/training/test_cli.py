from __future__ import annotations

from clauseforge.training.cli import main


def test_cli_dry_run(capsys: object) -> None:
    assert (
        main(
            [
                "--config",
                "training/configs/smoke.yaml",
                "--data",
                "data/processed/cuad/1.0.0-run-a",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "dry-run-valid" in output
    assert "test_split_access" in output
