"""CLI command for activation patching interpretability analysis."""

from __future__ import annotations

import argparse

from core.activation_patching_types import ActivationPatchingOptions
from serve.activation_patching_runner import run_activation_patching
from store.dataset_sdk import CrucibleClient


def run_activation_patching_command(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Handle activation-patch command invocation."""
    options = ActivationPatchingOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        clean_text=args.clean_text,
        corrupted_text=args.corrupted_text,
        target_token_index=args.target_token_index,
        base_model=getattr(args, "base_model", None),
        metric=args.metric,
    )
    run_activation_patching(options)
    return 0


def add_activation_patching_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register activation-patch subcommand."""
    parser = subparsers.add_parser(
        "activation-patch", help="Find causally important layers via activation patching",
    )
    parser.add_argument("--model-path", required=True, help="Model path or HF model ID")
    parser.add_argument("--clean-text", required=True, help="Clean input text")
    parser.add_argument("--corrupted-text", required=True, help="Corrupted input text")
    parser.add_argument("--output-dir", required=True, help="Directory for results JSON")
    parser.add_argument("--base-model", default=None, help="Base model for HF loading")
    parser.add_argument("--target-token-index", type=int, default=-1, help="Token index (-1 = last)")
    parser.add_argument("--metric", default="logit_diff", choices=["logit_diff", "prob"])
