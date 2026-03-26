"""Benchmark runner orchestrator.

This module coordinates running multiple benchmarks against a model
and aggregating results for comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.errors import CrucibleBenchmarkError, CrucibleDependencyError

if TYPE_CHECKING:
    from eval.benchmarks._model_loader import EvalModel


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


def _write_partial_results(
    output_path: str,
    model_path: str,
    results: list[BenchmarkResult],
    total_benchmarks: int,
) -> None:
    """Write incremental results so partial progress survives crashes."""
    avg = sum(r.score for r in results) / max(len(results), 1)
    data = {
        "status": "completed" if len(results) >= total_benchmarks else "partial",
        "job_type": "eval",
        "model_path": model_path,
        "average_score": round(avg, 2),
        "benchmarks_completed": len(results),
        "benchmarks_total": total_benchmarks,
        "benchmarks": [
            {"name": r.benchmark_name, "score": r.score,
             "num_examples": r.num_examples, "correct": r.correct}
            for r in results
        ],
    }
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def run_benchmarks(
    model_path: str,
    benchmarks: list[str],
    base_model_path: str | None = None,
    max_samples: int | None = None,
    output_path: str | None = None,
) -> EvaluationResult:
    """Run selected benchmarks against a model.

    Loads the model once and reuses it across all benchmarks. Writes
    incremental results to *output_path* after each benchmark so that
    partial progress is preserved if the process is killed.
    """
    from eval.benchmarks._model_loader import load_eval_model
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

    invalid = [b for b in benchmarks if b not in benchmark_map]
    valid = [b for b in benchmarks if b in benchmark_map]
    if not valid:
        raise CrucibleBenchmarkError(
            f"No valid benchmark names provided. Invalid: {invalid}. "
            f"Available: {list(benchmark_map)}"
        )

    # Load model once for all benchmarks
    print(f"CRUCIBLE_AGENT: Loading model for evaluation...", flush=True)
    eval_model = load_eval_model(model_path)
    print(f"CRUCIBLE_AGENT: Model loaded, running {len(valid)} benchmarks", flush=True)

    results: list[BenchmarkResult] = []
    for i, name in enumerate(valid, 1):
        print(f"CRUCIBLE_AGENT: [{i}/{len(valid)}] Running {name}...", flush=True)
        result = benchmark_map[name](model_path, max_samples=max_samples, eval_model=eval_model)
        results.append(result)
        print(f"CRUCIBLE_AGENT: [{i}/{len(valid)}] {name} done — score: {result.score}", flush=True)
        if output_path:
            _write_partial_results(output_path, model_path, results, len(valid))

    avg = sum(r.score for r in results) / max(len(results), 1)

    base_results: list[BenchmarkResult] = []
    if base_model_path:
        print(f"CRUCIBLE_AGENT: Loading base model for comparison...", flush=True)
        base_eval_model = load_eval_model(base_model_path)
        for i, name in enumerate(valid, 1):
            print(f"CRUCIBLE_AGENT: [base {i}/{len(valid)}] Running {name}...", flush=True)
            base_results.append(
                benchmark_map[name](base_model_path, max_samples=max_samples, eval_model=base_eval_model)
            )

    return EvaluationResult(
        model_path=model_path,
        benchmark_results=tuple(results),
        average_score=round(avg, 2),
        base_model_path=base_model_path,
        base_results=tuple(base_results),
    )
