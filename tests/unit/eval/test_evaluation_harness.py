"""Tests for evaluation harness."""

from __future__ import annotations

from pathlib import Path

from eval.evaluation_harness import EvaluationHarness


def test_evaluate_stores_result(tmp_path: Path) -> None:
    """Evaluation stores results to disk."""
    harness = EvaluationHarness(tmp_path)
    result = harness.evaluate("model.pt", benchmarks=["mmlu"])
    assert result.model_path == "model.pt"
    assert len(result.benchmark_results) == 1
    evals = harness.list_evaluations()
    assert len(evals) == 1


def test_list_empty_evaluations(tmp_path: Path) -> None:
    """Empty evaluations directory returns empty list."""
    harness = EvaluationHarness(tmp_path)
    assert harness.list_evaluations() == []


def test_evaluate_all_benchmarks(tmp_path: Path) -> None:
    """Running all benchmarks returns 7 results."""
    harness = EvaluationHarness(tmp_path)
    result = harness.evaluate("model.pt")
    assert len(result.benchmark_results) == 7


def test_load_stored_evaluation(tmp_path: Path) -> None:
    """Can load a stored evaluation result."""
    harness = EvaluationHarness(tmp_path)
    harness.evaluate("model.pt", benchmarks=["mmlu"])
    evals = harness.list_evaluations()
    loaded = harness.load_evaluation(evals[0])
    assert loaded["model_path"] == "model.pt"
