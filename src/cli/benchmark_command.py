"""Benchmark command wiring for Forge CLI.

This module isolates benchmark command parser and execution logic,
mapping CLI arguments to BenchmarkConfig for evaluation runs.
"""

from __future__ import annotations

import argparse

from core.benchmark_types import BenchmarkConfig
from serve.benchmark_runner import format_benchmark_report, run_benchmark
from store.dataset_sdk import ForgeClient


def run_benchmark_command(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    """Handle benchmark command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    config = BenchmarkConfig(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        version_id=args.version_id,
        max_token_length=args.max_token_length,
        batch_size=args.batch_size,
        run_perplexity=not args.no_perplexity,
        run_latency=not args.no_latency,
    )
    dataset = client.dataset(config.dataset_name)
    _, records = dataset.load_records(config.version_id)
    result = run_benchmark(
        records=records,
        config=config,
        data_root=client._config.data_root,
    )
    report = format_benchmark_report(result)
    print(report)
    return 0


def add_benchmark_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register benchmark subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate model with perplexity and latency benchmarks",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained model weights file",
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for benchmark report output",
    )
    parser.add_argument(
        "--version-id", default=None,
        help="Optional specific dataset version id",
    )
    parser.add_argument(
        "--max-token-length", type=int, default=512,
        help="Maximum token length per sequence",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--no-perplexity", action="store_true", default=False,
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--no-latency", action="store_true", default=False,
        help="Skip latency profiling",
    )
