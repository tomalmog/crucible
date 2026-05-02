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
    "mmlu", "hellaswag", "arc", "arc_easy", "winogrande", "truthfulqa",
    "gsm8k", "math", "bbh",
    "humaneval", "mbpp",
    "boolq", "piqa", "openbookqa",
)

# Our short names -> lm-eval task names.  Unmapped names pass through as-is.
_TASK_NAME_MAP: dict[str, str] = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "arc": "arc_challenge",
    "arc_easy": "arc_easy",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
    "math": "hendrycks_math",
    "truthfulqa": "truthfulqa_mc1",
    "bbh": "bbh",
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "boolq": "boolq",
    "piqa": "piqa",
    "openbookqa": "openbookqa",
}

_PREFERRED_METRIC: dict[str, str] = {
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "hendrycks_math": "exact_match,none",
    "truthfulqa_mc1": "acc,none",
    "bbh": "acc_norm,none",
    "humaneval": "pass@1,none",
    "mbpp": "pass@1,none",
    "boolq": "acc,none",
    "piqa": "acc_norm,none",
    "openbookqa": "acc_norm,none",
}


@dataclass(frozen=True)
class ModelComparisonResult:
    """Result from evaluating multiple models on the same benchmarks."""

    model_results: tuple["SingleModelResult", ...]
    benchmark_names: tuple[str, ...]


@dataclass(frozen=True)
class SingleModelResult:
    """Per-model result inside a comparison run."""

    model_path: str
    model_name: str
    average_score: float
    benchmark_results: tuple[BenchmarkResult, ...]


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


def _write_comparison_partial(
    output_path: str,
    completed: list["SingleModelResult"],
    total_models: int,
    benchmark_names: list[str],
) -> None:
    """Write incremental multi-model comparison results."""
    data = {
        "status": "completed" if len(completed) >= total_models else "partial",
        "job_type": "eval-compare",
        "models_completed": len(completed),
        "models_total": total_models,
        "benchmark_names": benchmark_names,
        "models": [
            {
                "model_path": m.model_path,
                "model_name": m.model_name,
                "average_score": m.average_score,
                "benchmarks": [
                    {"name": r.benchmark_name, "score": r.score,
                     "num_examples": r.num_examples, "correct": r.correct,
                     **({"error": r.details["error"]} if r.details.get("error") else {})}
                    for r in m.benchmark_results
                ],
            }
            for m in completed
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

    # Load the model once and reuse across all benchmarks
    try:
        model_obj, model_args = _load_model_once(model_path)
    except Exception as exc:
        print(f"CRUCIBLE_AGENT: Failed to load model: {exc}", flush=True)
        results = _all_failed(our_names, task_names, exc)
        if output_path:
            _write_partial_results(output_path, model_path, results, len(benchmarks))
        return EvaluationResult(
            model_path=model_path, benchmark_results=tuple(results),
            average_score=0.0, base_model_path=base_model_path,
        )

    results = _evaluate_model(
        model_obj, model_args, task_names, our_names, max_samples, output_path, model_path,
    )
    avg = sum(r.score for r in results) / max(len(results), 1)

    # Free primary model before loading base model
    del model_obj
    _gc_collect()

    base_results: list[BenchmarkResult] = []
    if base_model_path:
        print("CRUCIBLE_AGENT: Running base model comparison...", flush=True)
        try:
            base_obj, base_args = _load_model_once(base_model_path)
        except Exception as exc:
            print(f"CRUCIBLE_AGENT: Failed to load base model: {exc}", flush=True)
            base_results = []
        else:
            base_results = _evaluate_model(
                base_obj, base_args, task_names, our_names, max_samples, None, base_model_path,
            )
            del base_obj
            _gc_collect()

    return EvaluationResult(
        model_path=model_path,
        benchmark_results=tuple(results),
        average_score=round(avg, 2),
        base_model_path=base_model_path,
        base_results=tuple(base_results),
    )


def run_comparison(
    model_paths: list[str],
    benchmarks: list[str],
    max_samples: int | None = None,
    output_path: str | None = None,
) -> ModelComparisonResult:
    """Evaluate multiple models on the same benchmarks and aggregate results."""
    task_names = [_TASK_NAME_MAP.get(b, b) for b in benchmarks]
    our_names = list(benchmarks)
    completed: list[SingleModelResult] = []

    for idx, model_path in enumerate(model_paths, 1):
        model_name = _model_display_name(model_path)
        print(
            f"CRUCIBLE_AGENT: Model {idx}/{len(model_paths)}: {model_name}", flush=True,
        )
        try:
            model_obj, model_args = _load_model_once(model_path)
        except Exception as exc:
            print(f"CRUCIBLE_AGENT: Failed to load {model_name}: {exc}", flush=True)
            results = _all_failed(our_names, task_names, exc)
            completed.append(SingleModelResult(
                model_path=model_path,
                model_name=model_name,
                average_score=0.0,
                benchmark_results=tuple(results),
            ))
        else:
            results = _evaluate_model(
                model_obj, model_args, task_names, our_names, max_samples, None, model_path,
            )
            avg = sum(r.score for r in results) / max(len(results), 1)
            completed.append(SingleModelResult(
                model_path=model_path,
                model_name=model_name,
                average_score=round(avg, 2),
                benchmark_results=tuple(results),
            ))
            del model_obj
            _gc_collect()

        if output_path:
            _write_comparison_partial(output_path, completed, len(model_paths), our_names)

    return ModelComparisonResult(
        model_results=tuple(completed),
        benchmark_names=tuple(our_names),
    )


def _model_display_name(model_path: str) -> str:
    """Short human-readable name from a model path."""
    import os
    name = os.path.basename(model_path.rstrip("/"))
    return name if name else model_path


def _load_model_once(model_path: str) -> tuple[Any, str | None]:
    """Load the model once, return (model_object, model_args_string).

    For HF models: returns ("hf", model_args_string) so simple_evaluate
    can use its built-in HF loader.
    For Crucible .pt: returns (CrucibleLM instance, None).
    """
    resolved = _resolve_hf_path(model_path)
    if resolved:
        return "hf", f"pretrained={resolved}"
    from eval.crucible_lm_wrapper import CrucibleLM
    return CrucibleLM(model_path), None


def _is_generate_until_task(task_name: str) -> bool:
    """Check if a task uses generate_until by loading its config from lm-eval.

    Returns True if the task (or any subtask in a group) uses generate_until.
    Returns False for multiple_choice / loglikelihood tasks, or on any error.
    """
    try:
        _ensure_lm_eval()
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        task_dict = tm.load_task_or_group([task_name])

        def _walk(d: dict[str, Any]) -> bool:
            for v in d.values():
                if hasattr(v, "OUTPUT_TYPE") and v.OUTPUT_TYPE == "generate_until":
                    return True
                if isinstance(v, dict) and _walk(v):
                    return True
            return False

        return _walk(task_dict)
    except Exception:
        return False


def _evaluate_model(
    model_obj: Any,
    model_args: str | None,
    task_names: list[str],
    our_names: list[str],
    max_samples: int | None,
    output_path: str | None,
    model_path: str,
) -> list[BenchmarkResult]:
    """Run lm-eval one benchmark at a time to keep memory bounded.

    On macOS, generate_until benchmarks are skipped — they use a
    multiprocessing pool that crashes Python 3.13 with no catchable
    exception.  On Linux this is not an issue.
    """
    import sys
    is_macos = sys.platform == "darwin"

    results: list[BenchmarkResult] = []
    for i, (task_name, our_name) in enumerate(zip(task_names, our_names), 1):
        print(f"CRUCIBLE_AGENT: [{i}/{len(task_names)}] Running {our_name}...", flush=True)

        if is_macos and _is_generate_until_task(task_name):
            print(
                f"CRUCIBLE_AGENT: [{i}/{len(task_names)}] {our_name} skipped — "
                "this benchmark uses text generation which crashes on macOS "
                "due to a Python multiprocessing bug. "
                "Run on a remote Linux cluster instead.",
                flush=True,
            )
            result = BenchmarkResult(
                benchmark_name=our_name, score=0.0, num_examples=0, correct=0,
                details={"error": "crashes on macOS — run on a remote cluster"},
            )
        else:
            try:
                raw = _call_simple_evaluate(model_obj, model_args, [task_name], max_samples)
                result = _extract_benchmark_result(task_name, our_name, raw)
            except Exception as exc:
                print(f"CRUCIBLE_AGENT: [{i}/{len(task_names)}] {our_name} FAILED: {exc}", flush=True)
                result = BenchmarkResult(
                    benchmark_name=our_name, score=0.0, num_examples=0,
                    correct=0, details={"error": str(exc)},
                )

        results.append(result)
        print(f"CRUCIBLE_AGENT: [{i}/{len(task_names)}] {our_name} done — score: {result.score}", flush=True)
        if output_path:
            _write_partial_results(output_path, model_path, results, len(task_names))
        _gc_collect()
    return results


def _call_simple_evaluate(
    model_obj: Any,
    model_args: str | None,
    task_names: list[str],
    max_samples: int | None,
) -> dict[str, Any]:
    """Run lm_eval.simple_evaluate with a pre-loaded model."""
    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    _ensure_lm_eval()
    from lm_eval import simple_evaluate

    from eval.custom_benchmarks import get_benchmarks_include_path
    from core.config import CrucibleConfig

    # Load custom task configs if any exist
    include_path = get_benchmarks_include_path(CrucibleConfig.from_env().data_root)
    task_manager = None
    if include_path:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(include_path=include_path)

    needs_unsafe = "humaneval" in task_names
    kwargs: dict[str, Any] = {"tasks": task_names, "limit": max_samples}
    if task_manager is not None:
        kwargs["task_manager"] = task_manager
    if needs_unsafe:
        kwargs["confirm_run_unsafe_code"] = True

    if model_args is not None:
        # HF model: pass model type string + args
        kwargs["model"] = model_obj
        kwargs["model_args"] = model_args
        kwargs["batch_size"] = "auto"
        kwargs["device"] = _resolve_device()
    else:
        # CrucibleLM instance: pass directly
        kwargs["model"] = model_obj

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
    """Extract sample count from lm-eval output.

    For group tasks like MMLU, the top-level entry is None so we sum
    the effective counts from all subtasks that start with the group name.
    """
    for key in ("n-samples", "n_samples"):
        samples_map = raw.get(key) or {}
        n = samples_map.get(task_name)
        if isinstance(n, (int, float)):
            return int(n)
        if isinstance(n, dict):
            return int(n.get("effective", n.get("original", 0)))
        # Group task: sum subtask counts
        if n is None and samples_map:
            prefix = task_name + "_"
            total = 0
            for sub_key, sub_val in samples_map.items():
                if sub_key.startswith(prefix) and isinstance(sub_val, dict):
                    total += int(sub_val.get("effective", sub_val.get("original", 0)))
            if total > 0:
                return total
    return 0


def _all_failed(our_names: list[str], task_names: list[str], exc: Exception) -> list[BenchmarkResult]:
    """Return error results for every requested benchmark."""
    return [
        BenchmarkResult(benchmark_name=n, score=0.0, num_examples=0, correct=0, details={"error": str(exc)})
        for n in our_names
    ]


def _gc_collect() -> None:
    """Force garbage collection and free GPU cache between benchmarks."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _resolve_hf_path(model_path: str) -> str | None:
    """Resolve a model path to an HF-loadable path, or None for Crucible .pt.

    Handles: HF repo IDs (``gpt2``), local HF directories (with
    ``config.json``), and ``.pt`` files that have an adjacent
    ``hf_model/`` directory produced by trl training.
    """
    import os
    from serve.hf_model_loader import is_huggingface_model_id

    if is_huggingface_model_id(model_path):
        return model_path
    # .pt file with sibling hf_model/ directory (trl training output)
    if model_path.endswith(".pt") and os.path.isfile(model_path):
        hf_dir = os.path.join(os.path.dirname(model_path), "hf_model")
        if os.path.isdir(hf_dir) and os.path.exists(os.path.join(hf_dir, "config.json")):
            return hf_dir
    return None


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
