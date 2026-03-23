"""CLI command for activation PCA interpretability analysis."""

from __future__ import annotations

import argparse

from core.activation_pca_types import ActivationPcaOptions
from serve.activation_pca_runner import run_activation_pca
from store.dataset_sdk import CrucibleClient


def run_activation_pca_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle activation-pca command invocation."""
    options = ActivationPcaOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        base_model=getattr(args, "base_model", None),
        layer_index=args.layer_index,
        max_samples=args.max_samples,
        granularity=args.granularity,
        color_field=args.color_field or "",
    )
    dataset = client.dataset(options.dataset_name)
    _, records = dataset.load_records()
    run_activation_pca(options, records)
    return 0


def add_activation_pca_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register activation-pca subcommand."""
    parser = subparsers.add_parser(
        "activation-pca", help="Project layer activations to 2D via PCA",
    )
    parser.add_argument("--model-path", required=True, help="Model path or HF model ID")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Directory for results JSON")
    parser.add_argument("--base-model", default=None, help="Base model for HF loading")
    parser.add_argument("--layer-index", type=int, default=-1, help="Layer index (-1 = last)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max dataset samples")
    parser.add_argument("--granularity", default="sample", choices=["sample", "token"])
    parser.add_argument("--color-field", default="", help="Metadata field to color by")
