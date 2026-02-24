"""Compare command wiring for Forge CLI.

This module isolates the training run comparison command parser and
execution logic, mapping CLI arguments to the model comparison module.
"""

from __future__ import annotations

import argparse
from typing import Any

from core.errors import ForgeServeError
from serve.model_comparison import (
    compare_training_runs,
    format_comparison_report,
)
from store.dataset_sdk import ForgeClient


def add_compare_command(subparsers: Any) -> None:
    """Register compare subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "compare",
        help="Compare two training runs side by side",
    )
    parser.add_argument(
        "--run-a",
        required=True,
        help="First training run ID",
    )
    parser.add_argument(
        "--run-b",
        required=True,
        help="Second training run ID",
    )


def run_compare_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle compare command invocation.

    Args:
        client: SDK client for loading runs.
        args: Parsed CLI args with run_a and run_b.

    Returns:
        Exit code.
    """
    run_a = client.get_training_run(args.run_a)
    run_b = client.get_training_run(args.run_b)
    report = compare_training_runs(run_a, run_b)
    lines = format_comparison_report(report)
    for line in lines:
        print(line)
    return 0
