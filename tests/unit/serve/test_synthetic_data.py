"""Tests for synthetic data generation."""

from __future__ import annotations

from pathlib import Path
from serve.synthetic_data import export_synthetic_data, filter_by_quality, generate_synthetic_data, SyntheticExample


def test_generate_synthetic_data() -> None:
    examples = generate_synthetic_data(["What is AI?", "Explain ML"], count=5)
    assert len(examples) == 5
    assert all(e.prompt for e in examples)


def test_filter_by_quality() -> None:
    examples = [
        SyntheticExample("p1", "r1", 0.3),
        SyntheticExample("p2", "r2", 0.8),
        SyntheticExample("p3", "r3", 0.6),
    ]
    filtered = filter_by_quality(examples, 0.5)
    assert len(filtered) == 2


def test_export_synthetic_data(tmp_path: Path) -> None:
    examples = [SyntheticExample("p1", "r1", 0.8)]
    output = tmp_path / "synth.jsonl"
    count = export_synthetic_data(examples, str(output))
    assert count == 1
    assert output.exists()
