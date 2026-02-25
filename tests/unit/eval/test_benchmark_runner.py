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


def test_run_single_benchmark() -> None:
    """Run a single benchmark returns result."""
    result = run_benchmarks("test_model.pt", ["mmlu"])
    assert len(result.benchmark_results) == 1
    assert result.benchmark_results[0].benchmark_name == "mmlu"


def test_run_multiple_benchmarks() -> None:
    """Run multiple benchmarks returns all results."""
    result = run_benchmarks("test_model.pt", ["mmlu", "gsm8k", "arc"])
    assert len(result.benchmark_results) == 3


def test_run_with_base_model() -> None:
    """Running with base model produces comparison results."""
    result = run_benchmarks(
        "fine_tuned.pt", ["mmlu"], base_model_path="base.pt",
    )
    assert len(result.base_results) == 1
    assert result.base_model_path == "base.pt"


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
