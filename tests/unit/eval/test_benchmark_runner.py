"""Tests for benchmark runner."""

from __future__ import annotations

from eval.benchmark_runner import (
    AVAILABLE_BENCHMARKS,
    BenchmarkResult,
    run_benchmarks,
)


def test_available_benchmarks() -> None:
    """Available benchmarks list is populated."""
    assert len(AVAILABLE_BENCHMARKS) == 7
    assert "mmlu" in AVAILABLE_BENCHMARKS
    assert "humaneval" in AVAILABLE_BENCHMARKS
    assert "gsm8k" in AVAILABLE_BENCHMARKS


def test_benchmark_result_fields() -> None:
    """BenchmarkResult has expected fields."""
    br = BenchmarkResult(
        benchmark_name="test", score=75.5,
        num_examples=100, correct=75,
    )
    assert br.score == 75.5
    assert br.correct == 75


def test_unknown_benchmark_skipped() -> None:
    """Unknown benchmark names are skipped."""
    result = run_benchmarks("model.pt", ["nonexistent"])
    assert len(result.benchmark_results) == 0
