"""Model merge command wiring for Forge CLI.

This module provides the merge command for combining
multiple model weights into a single model.
"""

from __future__ import annotations

import argparse

from serve.model_merger import SUPPORTED_MERGE_METHODS, MergeConfig, merge_models
from store.dataset_sdk import ForgeClient


def run_merge_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle merge command invocation."""
    weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else ()
    config = MergeConfig(
        model_paths=tuple(args.models),
        method=args.method,
        weights=weights,
        output_path=args.output,
    )
    result = merge_models(config)
    print(f"output_path={result.output_path}")
    print(f"method={result.method}")
    print(f"num_models={result.num_models}")
    print(f"num_parameters={result.num_parameters}")
    return 0


def add_merge_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register merge subcommand."""
    parser = subparsers.add_parser("merge", help="Merge multiple model weights")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to models to merge")
    parser.add_argument(
        "--method", default="average",
        choices=list(SUPPORTED_MERGE_METHODS),
        help="Merge method (slerp, ties, dare, average)",
    )
    parser.add_argument("--weights", default=None, help="Comma-separated per-model weights")
    parser.add_argument("--output", default="./merged_model.pt", help="Output path for merged model")
