"""Evaluation command wiring for Crucible CLI.

This module provides the eval command for running standardized
benchmarks against trained models.
"""

from __future__ import annotations

import argparse
import json

from eval.evaluation_harness import EvaluationHarness
from eval.benchmark_runner import AVAILABLE_BENCHMARKS, run_comparison
from store.dataset_sdk import CrucibleClient


def run_eval_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle eval command invocation."""
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None

    # Multi-model comparison mode
    model_paths = getattr(args, "model_paths", None)
    if model_paths:
        paths = [p.strip() for p in model_paths.split(",") if p.strip()]
        if benchmarks is None:
            benchmarks = list(AVAILABLE_BENCHMARKS)
        result = run_comparison(paths, benchmarks, max_samples=args.max_samples)
        print(json.dumps({
            "status": "completed",
            "job_type": "eval-compare",
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
                for m in result.model_results
            ],
        }), flush=True)
        return 0

    # Single-model mode (backwards compat)
    harness = EvaluationHarness(client._config.data_root)
    result = harness.evaluate(
        model_path=args.model_path,
        benchmarks=benchmarks,
        base_model_path=args.base_model,
        max_samples=args.max_samples,
    )
    print(f"model_path={result.model_path}")
    print(f"average_score={result.average_score}")
    for br in result.benchmark_results:
        line = f"benchmark={br.benchmark_name}  score={br.score}  examples={br.num_examples}  correct={br.correct}"
        if br.details.get("error"):
            line += f"  error={br.details['error']}"
        print(line)
    if result.base_results:
        print(f"\nbase_model_path={result.base_model_path}")
        for br in result.base_results:
            print(f"base_benchmark={br.benchmark_name}  score={br.score}")
    return 0


def add_eval_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register eval subcommand."""
    parser = subparsers.add_parser("eval", help="Run evaluation benchmarks against a model")
    # Single-model (backwards compat)
    parser.add_argument("--model-path", default=None, help="Path to the model to evaluate")
    # Multi-model comparison
    parser.add_argument(
        "--model-paths",
        default=None,
        help="Comma-separated list of model paths for comparison evaluation",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help=f"Comma-separated list of benchmarks ({','.join(AVAILABLE_BENCHMARKS)})",
    )
    parser.add_argument("--base-model", help="Optional base model path for comparison (single-model mode only)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max examples per benchmark")
