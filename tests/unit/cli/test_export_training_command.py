"""Unit tests for CLI export-training command."""

from __future__ import annotations

from cli.main import main


def test_cli_export_training_prints_manifest_path(tmp_path, capsys) -> None:
    """CLI export-training should print the generated manifest path."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("alpha", encoding="utf-8")
    data_root = tmp_path / "crucible"

    ingest_args = [
        "--data-root",
        str(data_root),
        "ingest",
        str(source_dir),
        "--dataset",
        "demo",
    ]
    export_args = [
        "--data-root",
        str(data_root),
        "export-training",
        "--dataset",
        "demo",
        "--output-dir",
        str(tmp_path / "exports"),
        "--shard-size",
        "1",
    ]
    main(ingest_args)

    exit_code = main(export_args)
    output = capsys.readouterr().out.strip().splitlines()[-1]

    assert exit_code == 0 and output.endswith("training_manifest.json")
