"""Evaluation harness for comprehensive model assessment.

This module provides a high-level interface for running standardized
benchmarks and storing results alongside training runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval.benchmark_runner import (
    AVAILABLE_BENCHMARKS,
    BenchmarkResult,
    EvaluationResult,
    run_benchmarks,
)


class EvaluationHarness:
    """High-level evaluation harness for model assessment."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._eval_dir = data_root / "evaluations"

    def evaluate(
        self,
        model_path: str,
        benchmarks: list[str] | None = None,
        base_model_path: str | None = None,
        max_samples: int | None = None,
    ) -> EvaluationResult:
        """Run evaluation benchmarks and store results."""
        if benchmarks is None:
            benchmarks = list(AVAILABLE_BENCHMARKS)
        result = run_benchmarks(model_path, benchmarks, base_model_path, max_samples=max_samples)
        self._store_result(result)
        return result

    def list_evaluations(self) -> list[str]:
        """List all stored evaluation result files."""
        if not self._eval_dir.exists():
            return []
        return sorted(f.stem for f in self._eval_dir.glob("*.json"))

    def load_evaluation(self, eval_id: str) -> dict[str, Any]:
        """Load a stored evaluation result."""
        path = self._eval_dir / f"{eval_id}.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _store_result(self, result: EvaluationResult) -> None:
        """Store evaluation results as JSON."""
        self._eval_dir.mkdir(parents=True, exist_ok=True)
        import hashlib
        import time
        eval_id = hashlib.sha256(
            f"{result.model_path}{time.time()}".encode()
        ).hexdigest()[:12]
        path = self._eval_dir / f"{eval_id}.json"
        data = {
            "model_path": result.model_path,
            "average_score": result.average_score,
            "benchmarks": [
                {
                    "name": r.benchmark_name,
                    "score": r.score,
                    "num_examples": r.num_examples,
                    "correct": r.correct,
                    "details": r.details,
                }
                for r in result.benchmark_results
            ],
            "base_model_path": result.base_model_path,
            "base_benchmarks": [
                {
                    "name": r.benchmark_name,
                    "score": r.score,
                    "num_examples": r.num_examples,
                    "correct": r.correct,
                    "details": r.details,
                }
                for r in result.base_results
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
