"""Benchmark runner — delegates to lm-evaluation-harness (lm_eval)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.errors import CrucibleBenchmarkError, CrucibleDependencyError


@dataclass(frozen=True)
class BenchmarkResult:
    """Result from running a single benchmark (score is 0-100)."""

    benchmark_name: str
    score: float
    num_examples: int
    correct: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated results from running multiple benchmarks."""

    model_path: str
    benchmark_results: tuple[BenchmarkResult, ...]
    average_score: float
    base_model_path: str | None = None
    base_results: tuple[BenchmarkResult, ...] = ()


AVAILABLE_BENCHMARKS = (
    "mmlu", "humaneval", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande",
)

# Our short names -> lm-eval task names.  Unmapped names pass through as-is.
_TASK_NAME_MAP: dict[str, str] = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "arc": "arc_challenge",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
    "truthfulqa": "truthfulqa_mc1",
    "humaneval": "humaneval",
}

_PREFERRED_METRIC: dict[str, str] = {
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc1": "acc,none",
    "humaneval": "pass@1,none",
}


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
             "num_examples": r.num_examples, "correct": r.correct,
             **({"error": r.details["error"]} if r.details.get("error") else {})}
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
    """Run selected benchmarks via lm-evaluation-harness."""
    task_names = [_TASK_NAME_MAP.get(b, b) for b in benchmarks]
    our_names = list(benchmarks)
    print(f"CRUCIBLE_AGENT: Running {len(benchmarks)} benchmarks via lm-eval...", flush=True)

    results = _evaluate_model(model_path, task_names, our_names, max_samples)
    if output_path:
        _write_partial_results(output_path, model_path, results, len(benchmarks))
    avg = sum(r.score for r in results) / max(len(results), 1)
    base_results = _run_base_comparison(base_model_path, task_names, our_names, max_samples)

    return EvaluationResult(
        model_path=model_path,
        benchmark_results=tuple(results),
        average_score=round(avg, 2),
        base_model_path=base_model_path,
        base_results=tuple(base_results),
    )


def _evaluate_model(
    model_path: str,
    task_names: list[str],
    our_names: list[str],
    max_samples: int | None,
) -> list[BenchmarkResult]:
    """Run lm-eval against *model_path* and return per-task results."""
    try:
        raw = _call_simple_evaluate(model_path, task_names, max_samples)
    except Exception as exc:
        print(f"CRUCIBLE_AGENT: lm-eval evaluation failed: {exc}", flush=True)
        return _all_failed(our_names, task_names, exc)

    results: list[BenchmarkResult] = []
    for our_name, task_name in zip(our_names, task_names):
        result = _extract_benchmark_result(task_name, our_name, raw)
        results.append(result)
        print(
            f"CRUCIBLE_AGENT: {our_name} done — score: {result.score}",
            flush=True,
        )
    return results


def _call_simple_evaluate(
    model_path: str,
    task_names: list[str],
    max_samples: int | None,
) -> dict[str, Any]:
    """Dispatch to lm_eval.simple_evaluate for the right model type."""
    _ensure_lm_eval()
    from lm_eval import simple_evaluate
    from serve.hf_model_loader import is_huggingface_model_id

    needs_unsafe = "humaneval" in task_names
    if is_huggingface_model_id(model_path):
        return _evaluate_hf_model(simple_evaluate, model_path, task_names, max_samples, needs_unsafe)
    return _evaluate_crucible_model(simple_evaluate, model_path, task_names, max_samples, needs_unsafe)


def _evaluate_hf_model(
    simple_evaluate: Any, model_path: str, task_names: list[str],
    max_samples: int | None, needs_unsafe: bool,
) -> dict[str, Any]:
    """Run lm-eval with the built-in hf model type."""
    kwargs: dict[str, Any] = {
        "model": "hf", "model_args": f"pretrained={model_path}",
        "tasks": task_names, "limit": max_samples, "batch_size": "auto",
        "device": _resolve_device(),
    }
    if needs_unsafe:
        kwargs["confirm_run_unsafe_code"] = True
    return simple_evaluate(**kwargs)


def _evaluate_crucible_model(
    simple_evaluate: Any, model_path: str, task_names: list[str],
    max_samples: int | None, needs_unsafe: bool,
) -> dict[str, Any]:
    """Run lm-eval with the CrucibleLM wrapper."""
    from eval.crucible_lm_wrapper import CrucibleLM

    lm = CrucibleLM(model_path)
    kwargs: dict[str, Any] = {"model": lm, "tasks": task_names, "limit": max_samples}
    if needs_unsafe:
        kwargs["confirm_run_unsafe_code"] = True
    return simple_evaluate(**kwargs)


def _extract_benchmark_result(
    task_name: str, our_name: str, raw: dict[str, Any],
) -> BenchmarkResult:
    """Convert one task's lm-eval output into a BenchmarkResult."""
    task_results = (raw.get("results") or {}).get(task_name)
    if task_results is None:
        return BenchmarkResult(
            benchmark_name=our_name, score=0.0, num_examples=0, correct=0,
            details={"error": f"No results returned for task '{task_name}'"},
        )
    score_frac = _pick_primary_metric(task_name, task_results)
    score = round(score_frac * 100, 2)
    num_examples = _get_num_examples(task_name, raw)
    correct = round(score_frac * num_examples)
    return BenchmarkResult(
        benchmark_name=our_name, score=score, num_examples=num_examples,
        correct=correct, details=dict(task_results),
    )


def _pick_primary_metric(task_name: str, task_results: dict[str, Any]) -> float:
    """Choose the best metric from the task results dict."""
    preferred = _PREFERRED_METRIC.get(task_name)
    if preferred and preferred in task_results:
        return float(task_results[preferred])
    for key in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                "pass@1,none", "exact_match,none"):
        if key in task_results:
            return float(task_results[key])
    return 0.0


def _get_num_examples(task_name: str, raw: dict[str, Any]) -> int:
    """Extract sample count from lm-eval output."""
    for key in ("n-samples", "n_samples"):
        n = (raw.get(key) or {}).get(task_name)
        if isinstance(n, (int, float)):
            return int(n)
        if isinstance(n, dict):
            return int(n.get("effective", n.get("original", 0)))
    return 0


def _all_failed(our_names: list[str], task_names: list[str], exc: Exception) -> list[BenchmarkResult]:
    """Return error results for every requested benchmark."""
    return [
        BenchmarkResult(benchmark_name=n, score=0.0, num_examples=0, correct=0, details={"error": str(exc)})
        for n in our_names
    ]


def _run_base_comparison(
    base_model_path: str | None, task_names: list[str],
    our_names: list[str], max_samples: int | None,
) -> list[BenchmarkResult]:
    """Evaluate the base model if a path was provided."""
    if not base_model_path:
        return []
    print("CRUCIBLE_AGENT: Running base model comparison...", flush=True)
    return _evaluate_model(base_model_path, task_names, our_names, max_samples)


def _resolve_device() -> str:
    """Pick the best available device for evaluation."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS has compatibility issues with lm-eval; use CPU
    return "cpu"


def _ensure_lm_eval() -> None:
    """Verify that lm-evaluation-harness is installed."""
    try:
        import lm_eval  # noqa: F401
    except ImportError as error:
        raise CrucibleDependencyError(
            "Evaluation benchmarks require lm-evaluation-harness. "
            "Install with: pip install lm-eval"
        ) from error
