"""Replay command wiring for Forge CLI.

This module isolates replay command parser and execution logic.
It keeps the top-level CLI module focused and within size constraints.
"""

from __future__ import annotations

import argparse
from typing import Any

from core.types import TrainingRunResult
from serve.reproducibility_replay import replay_training_run
from store.dataset_sdk import ForgeClient


def add_replay_command(subparsers: Any) -> None:
    """Register replay subcommand.

    Args:
        subparsers: Argparse subparsers group from build_parser.
    """
    parser = subparsers.add_parser(
        "replay",
        help="Replay a training run from a reproducibility bundle",
    )
    parser.add_argument(
        "--bundle-path",
        required=True,
        help="Path to a reproducibility_bundle.json file",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for training output directory",
    )


def run_replay_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle replay command invocation.

    Args:
        client: ForgeClient SDK instance.
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    result = replay_training_run(
        client=client,
        bundle_path=args.bundle_path,
        output_dir=args.output_dir,
    )
    _print_replay_result(result)
    return 0


def _print_replay_result(result: TrainingRunResult) -> None:
    """Print replay result summary to stdout."""
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"run_id={result.run_id or '-'}")
