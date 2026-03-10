"""Benchmark runner orchestrator.

This module coordinates running multiple benchmarks against a model
and aggregating results for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.errors import CrucibleDependencyError


@dataclass(frozen=True)
class BenchmarkResult:
    """Result from running a single benchmark.

    Attributes:
        benchmark_name: Name of the benchmark.
        score: Overall score (0-100).
        num_examples: Number of evaluation examples.
        correct: Number of correct answers.
        details: Additional benchmark-specific metrics.
    """

    benchmark_name: str
    score: float
    num_examples: int
    correct: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated results from running multiple benchmarks.

    Attributes:
        model_path: Path to the evaluated model.
        benchmark_results: Results per benchmark.
        average_score: Mean score across all benchmarks.
        base_model_path: Optional base model for comparison.
        base_results: Base model results if comparison was run.
    """

    model_path: str
    benchmark_results: tuple[BenchmarkResult, ...]
    average_score: float
    base_model_path: str | None = None
    base_results: tuple[BenchmarkResult, ...] = ()


AVAILABLE_BENCHMARKS = (
    "mmlu", "humaneval", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande",
)


def run_benchmarks(
    model_path: str,
    benchmarks: list[str],
    base_model_path: str | None = None,
    max_samples: int | None = None,
) -> EvaluationResult:
    """Run selected benchmarks against a model."""
    from eval.benchmarks.mmlu import run_mmlu
    from eval.benchmarks.humaneval import run_humaneval
    from eval.benchmarks.gsm8k import run_gsm8k
    from eval.benchmarks.hellaswag import run_hellaswag
    from eval.benchmarks.arc import run_arc
    from eval.benchmarks.truthfulqa import run_truthfulqa
    from eval.benchmarks.winogrande import run_winogrande

    benchmark_map = {
        "mmlu": run_mmlu,
        "humaneval": run_humaneval,
        "gsm8k": run_gsm8k,
        "hellaswag": run_hellaswag,
        "arc": run_arc,
        "truthfulqa": run_truthfulqa,
        "winogrande": run_winogrande,
    }
    results: list[BenchmarkResult] = []
    for name in benchmarks:
        if name not in benchmark_map:
            continue
        result = benchmark_map[name](model_path, max_samples=max_samples)
        results.append(result)
    avg = sum(r.score for r in results) / max(len(results), 1)
    base_results: list[BenchmarkResult] = []
    if base_model_path:
        for name in benchmarks:
            if name not in benchmark_map:
                continue
            base_results.append(benchmark_map[name](base_model_path, max_samples=max_samples))
    return EvaluationResult(
        model_path=model_path,
        benchmark_results=tuple(results),
        average_score=round(avg, 2),
        base_model_path=base_model_path,
        base_results=tuple(base_results),
    )
