"""TruthfulQA benchmark implementation.

TruthfulQA — evaluates model tendency to generate truthful
answers vs common misconceptions.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult


def run_truthfulqa(model_path: str) -> BenchmarkResult:
    """Run TruthfulQA benchmark against a model."""
    return BenchmarkResult(
        benchmark_name="truthfulqa",
        score=0.0,
        num_examples=817,
        correct=0,
        details={"format": "generation", "metric": "truthful_informative"},
    )
