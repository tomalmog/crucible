"""Unit tests for CLI command handling."""

from __future__ import annotations

from dataclasses import replace

from cli.main import main
from core.config import CrucibleConfig


def test_cli_ingest_creates_version(tmp_path, capsys) -> None:
    """CLI ingest should print the created version id."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    source_path = "tests/fixtures/raw/local_a.txt"
    args = [
        "--data-root",
        str(config.data_root),
        "ingest",
        source_path,
        "--dataset",
        "cli-demo",
    ]

    exit_code = main(args)
    output = capsys.readouterr().out.strip()

    assert exit_code == 0 and bool(output)
