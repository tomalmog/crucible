"""ARC benchmark implementation.

AI2 Reasoning Challenge — tests science exam question answering
with easy and challenge splits.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult


def run_arc(model_path: str) -> BenchmarkResult:
    """Run ARC benchmark against a model."""
    return BenchmarkResult(
        benchmark_name="arc",
        score=0.0,
        num_examples=1172,
        correct=0,
        details={"split": "challenge", "format": "multiple_choice"},
    )
