"""Run-spec CLI command wiring.

This module registers the run-spec subcommand and delegates execution to the
shared run-spec engine used by CLI and SDK entry points.
"""

from __future__ import annotations

import argparse

from core.run_spec_execution import execute_run_spec_file
from store.dataset_sdk import CrucibleClient


def add_run_spec_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register run-spec subcommand."""
    parser = subparsers.add_parser(
        "run-spec",
        help="Run a declarative YAML pipeline spec",
    )
    parser.add_argument("spec_file", help="Path to YAML run-spec file")


def run_run_spec_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle run-spec command invocation."""
    output_lines = execute_run_spec_file(client, args.spec_file)
    for line in output_lines:
        print(line)
    return 0
