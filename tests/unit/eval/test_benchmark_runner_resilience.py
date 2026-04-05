"""Tests for benchmark_runner resilience with lm-eval-harness backend.

Validates error handling, partial results, and result structure when
benchmarks are delegated to lm_eval.simple_evaluate().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from eval.benchmark_runner import (
    BenchmarkResult,
    run_benchmarks,
    _write_partial_results,
    _extract_benchmark_result,
)


def _fake_lm_eval_output(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a dict shaped like lm_eval.simple_evaluate() output."""
    n_samples = {task: 100 for task in task_results}
    return {"results": task_results, "n-samples": n_samples}


def _patch_load():
    """Mock _load_model_once so tests skip real model loading."""
    return patch("eval.benchmark_runner._load_model_once", return_value=("hf", "pretrained=fake"))


def _patch_eval(side_effect_or_value: Any):
    """Mock _call_simple_evaluate. Exceptions are raised; dicts are returned."""
    if isinstance(side_effect_or_value, BaseException):
        return patch("eval.benchmark_runner._call_simple_evaluate", side_effect=side_effect_or_value)
    if callable(side_effect_or_value):
        return patch("eval.benchmark_runner._call_simple_evaluate", side_effect=side_effect_or_value)
    return patch("eval.benchmark_runner._call_simple_evaluate", return_value=side_effect_or_value)


# ── Per-benchmark error isolation ────────────────────────────────────────


def test_evaluation_failure_records_all_zero() -> None:
    """If simple_evaluate raises, all benchmarks get score=0 with error."""
    with _patch_load(), _patch_eval(RuntimeError("OOM")):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k"])

    assert len(result.benchmark_results) == 2
    assert all(r.score == 0.0 for r in result.benchmark_results)
    assert all("OOM" in r.details["error"] for r in result.benchmark_results)


def test_missing_task_in_results_records_zero() -> None:
    """A task missing from lm-eval output gets score=0 with error detail."""
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.75}})
    result = _extract_benchmark_result("gsm8k", "gsm8k", raw)
    assert result.score == 0.0
    assert "error" in result.details


# ── Average score correctness ────────────────────────────────────────────


def test_average_score_computed_correctly() -> None:
    """Average is the mean of all benchmark scores."""
    raw_mmlu = _fake_lm_eval_output({"mmlu": {"acc,none": 0.80}})
    raw_hella = _fake_lm_eval_output({"hellaswag": {"acc_norm,none": 0.60}})
    calls = iter([raw_mmlu, raw_hella])
    with _patch_load(), patch("eval.benchmark_runner._call_simple_evaluate", side_effect=lambda *a: next(calls)):
        result = run_benchmarks("model.pt", ["mmlu", "hellaswag"])

    assert result.average_score == 70.0


def test_single_benchmark_average_equals_score() -> None:
    """Single benchmark: average equals that benchmark's score."""
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.835}})
    with _patch_load(), _patch_eval(raw):
        result = run_benchmarks("model.pt", ["mmlu"])

    assert result.average_score == 83.5


def test_all_failed_average_is_zero() -> None:
    """If evaluation raises, average is 0."""
    with _patch_load(), _patch_eval(RuntimeError("dead")):
        result = run_benchmarks("model.pt", ["mmlu"])

    assert result.average_score == 0.0


# ── Partial result writing ───────────────────────────────────────────────


def test_partial_results_written_on_success(tmp_path: Path) -> None:
    """Partial results file is written after evaluation."""
    output = tmp_path / "r.json"
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.90}})
    with _patch_load(), _patch_eval(raw):
        run_benchmarks("model.pt", ["mmlu"], output_path=str(output))

    assert output.exists()
    import json
    data = json.loads(output.read_text())
    assert data["status"] == "completed"
    assert len(data["benchmarks"]) == 1


def test_write_partial_results_format(tmp_path: Path) -> None:
    """_write_partial_results produces correct JSON structure."""
    path = str(tmp_path / "partial.json")
    results = [
        BenchmarkResult(benchmark_name="mmlu", score=80.0, num_examples=100, correct=80),
        BenchmarkResult(benchmark_name="gsm8k", score=60.0, num_examples=50, correct=30),
    ]
    _write_partial_results(path, "model.pt", results, 3)

    import json
    data = json.loads(Path(path).read_text())
    assert data["status"] == "partial"
    assert data["benchmarks_completed"] == 2
    assert data["benchmarks_total"] == 3


# ── Base model comparison ────────────────────────────────────────────────


def test_base_model_results_populated() -> None:
    """Base model benchmarks are returned in base_results."""
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.75}})
    with _patch_load(), _patch_eval(raw):
        result = run_benchmarks("model.pt", ["mmlu"], base_model_path="base.pt")

    assert result.base_model_path == "base.pt"
    assert len(result.base_results) == 1
    assert result.base_results[0].benchmark_name == "mmlu"


def test_base_model_failure_does_not_affect_primary() -> None:
    """Primary results are preserved even if base model loading fails."""
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.90}})
    load_count = [0]

    def mock_load(path):
        load_count[0] += 1
        if load_count[0] > 1:
            raise RuntimeError("base model dead")
        return ("hf", "pretrained=fake")

    with patch("eval.benchmark_runner._load_model_once", side_effect=mock_load), _patch_eval(raw):
        result = run_benchmarks("model.pt", ["mmlu"], base_model_path="base.pt")

    assert result.benchmark_results[0].score == 90.0
    # Base model failed to load — no base results
    assert len(result.base_results) == 0


# ── EvaluationResult structure ───────────────────────────────────────────


def test_result_contains_model_path() -> None:
    """model_path is preserved in the result."""
    raw = _fake_lm_eval_output({"mmlu": {"acc,none": 0.5}})
    with _patch_load(), _patch_eval(raw):
        result = run_benchmarks("/my/model.pt", ["mmlu"])

    assert result.model_path == "/my/model.pt"


def test_task_name_mapping() -> None:
    """Our short names are mapped to lm-eval task names correctly."""
    from eval.benchmark_runner import _TASK_NAME_MAP
    assert _TASK_NAME_MAP["arc"] == "arc_challenge"
    assert _TASK_NAME_MAP["truthfulqa"] == "truthfulqa_mc1"
    assert _TASK_NAME_MAP["mmlu"] == "mmlu"


def test_extract_benchmark_result_prefers_acc_norm() -> None:
    """Preferred metric is used when available."""
    raw = _fake_lm_eval_output({
        "hellaswag": {"acc,none": 0.60, "acc_norm,none": 0.70},
    })
    result = _extract_benchmark_result("hellaswag", "hellaswag", raw)
    assert result.score == 70.0  # acc_norm preferred for hellaswag
