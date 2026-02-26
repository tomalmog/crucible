"""Evaluation command wiring for Forge CLI.

This module provides the eval command for running standardized
benchmarks against trained models.
"""

from __future__ import annotations

import argparse
import json

from eval.evaluation_harness import EvaluationHarness
from eval.benchmark_runner import AVAILABLE_BENCHMARKS
from store.dataset_sdk import ForgeClient


def run_eval_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle eval command invocation."""
    harness = EvaluationHarness(client._config.data_root)
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    result = harness.evaluate(
        model_path=args.model_path,
        benchmarks=benchmarks,
        base_model_path=args.base_model,
    )
    print(f"model_path={result.model_path}")
    print(f"average_score={result.average_score}")
    for br in result.benchmark_results:
        print(f"benchmark={br.benchmark_name}  score={br.score}  examples={br.num_examples}  correct={br.correct}")
    if result.base_results:
        print(f"\nbase_model_path={result.base_model_path}")
        for br in result.base_results:
            print(f"base_benchmark={br.benchmark_name}  score={br.score}")
    return 0


def add_eval_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register eval subcommand."""
    parser = subparsers.add_parser("eval", help="Run evaluation benchmarks against a model")
    parser.add_argument("--model-path", required=True, help="Path to the model to evaluate")
    parser.add_argument(
        "--benchmarks",
        default=None,
        help=f"Comma-separated list of benchmarks ({','.join(AVAILABLE_BENCHMARKS)})",
    )
    parser.add_argument("--base-model", help="Optional base model path for comparison")
