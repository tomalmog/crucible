"""Tests for benchmark runner."""

from __future__ import annotations

from eval.benchmark_runner import (
    AVAILABLE_BENCHMARKS,
    BenchmarkResult,
    run_benchmarks,
)


def test_available_benchmarks() -> None:
    """Available benchmarks list is populated."""
    assert len(AVAILABLE_BENCHMARKS) == 15
    assert "mmlu" in AVAILABLE_BENCHMARKS
    assert "humaneval" in AVAILABLE_BENCHMARKS
    assert "gsm8k" in AVAILABLE_BENCHMARKS
    assert "gpqa" in AVAILABLE_BENCHMARKS
    assert "bbh" in AVAILABLE_BENCHMARKS


def test_benchmark_result_fields() -> None:
    """BenchmarkResult has expected fields."""
    br = BenchmarkResult(
        benchmark_name="test", score=75.5,
        num_examples=100, correct=75,
    )
    assert br.score == 75.5
    assert br.correct == 75


def test_unknown_benchmark_returns_error_result() -> None:
    """Unknown benchmark names produce BenchmarkResult with error details."""
    # lm-eval passes unknown names through; the error surfaces in the result
    result = run_benchmarks("model.pt", ["nonexistent"])
    assert len(result.benchmark_results) == 1
    assert result.benchmark_results[0].score == 0.0
    assert result.benchmark_results[0].details.get("error")
