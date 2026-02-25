"""GSM8K benchmark implementation.

Grade School Math 8K — tests mathematical reasoning with
8.5K grade school math word problems.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult


def run_gsm8k(model_path: str) -> BenchmarkResult:
    """Run GSM8K benchmark against a model."""
    return BenchmarkResult(
        benchmark_name="gsm8k",
        score=0.0,
        num_examples=1319,
        correct=0,
        details={"format": "chain_of_thought", "metric": "exact_match"},
    )
