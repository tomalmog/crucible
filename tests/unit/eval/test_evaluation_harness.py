"""Tests for evaluation harness."""

from __future__ import annotations

from pathlib import Path

from eval.evaluation_harness import EvaluationHarness


def test_list_empty_evaluations(tmp_path: Path) -> None:
    """Empty evaluations directory returns empty list."""
    harness = EvaluationHarness(tmp_path)
    assert harness.list_evaluations() == []


