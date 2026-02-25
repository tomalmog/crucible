"""End-to-end integration tests for synthetic data generation workflow.

Tests cover the synthetic data generation pipeline including
generation, quality filtering, JSONL export, and CLI entry points.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.main import main
from serve.synthetic_data import (
    SyntheticExample,
    export_synthetic_data,
    filter_by_quality,
    generate_synthetic_data,
)


def test_generate_produces_examples() -> None:
    """Generating from two seed prompts with count=10 should yield 10 examples."""
    result = generate_synthetic_data(["prompt1", "prompt2"], count=10)

    assert len(result) == 10
    for ex in result:
        assert isinstance(ex, SyntheticExample)
        assert ex.prompt
        assert ex.response


def test_filter_by_quality() -> None:
    """Filtering 20 examples at min_quality=0.75 keeps only high-quality ones."""
    examples = generate_synthetic_data(["prompt1", "prompt2"], count=20)

    filtered = filter_by_quality(examples, min_quality=0.75)

    assert len(filtered) > 0
    assert len(filtered) <= len(examples)
    for ex in filtered:
        assert ex.quality_score >= 0.75


def test_export_creates_jsonl(tmp_path: Path) -> None:
    """Exporting 5 examples should create a JSONL file with valid JSON lines."""
    examples = generate_synthetic_data(["seed prompt"], count=5)
    output_path = tmp_path / "exported.jsonl"

    count = export_synthetic_data(examples, str(output_path))

    assert count == 5
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        obj = json.loads(line)
        assert "prompt" in obj
        assert "response" in obj


def test_full_pipeline(tmp_path: Path) -> None:
    """Generate, filter, export, then read back should match filtered count."""
    examples = generate_synthetic_data(["alpha", "beta", "gamma"], count=15)
    filtered = filter_by_quality(examples, min_quality=0.75)
    output_path = tmp_path / "pipeline_output.jsonl"

    exported_count = export_synthetic_data(filtered, str(output_path))

    assert exported_count == len(filtered)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(filtered)
    for line in lines:
        obj = json.loads(line)
        assert obj["quality_score"] >= 0.75


def test_cli_pipeline(
    tmp_path: Path,
    seed_prompts_file: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI synthetic command should exit 0 and report exported count."""
    output_file = str(tmp_path / "out.jsonl")

    exit_code = main([
        "--data-root", str(tmp_path),
        "synthetic",
        "--seed-prompts", seed_prompts_file,
        "--count", "5",
        "--output", output_file,
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "exported=" in captured
    assert Path(output_file).exists()


def test_cli_missing_seed_file(tmp_path: Path) -> None:
    """CLI with a nonexistent seed file should return a non-zero exit code."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "synthetic",
        "--seed-prompts", "/nonexistent/file.txt",
        "--count", "5",
    ])

    assert exit_code != 0
