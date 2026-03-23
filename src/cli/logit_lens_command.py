"""CLI command for logit lens interpretability analysis."""

from __future__ import annotations

import argparse

from core.logit_lens_types import LogitLensOptions
from serve.logit_lens_runner import run_logit_lens
from store.dataset_sdk import CrucibleClient


def run_logit_lens_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle logit-lens command invocation."""
    options = LogitLensOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        input_text=args.input_text,
        base_model=getattr(args, "base_model", None),
        top_k=args.top_k,
        layer_indices=args.layer_indices or "",
    )
    run_logit_lens(options)
    return 0


def add_logit_lens_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register logit-lens subcommand."""
    parser = subparsers.add_parser(
        "logit-lens", help="Project hidden states through unembedding per layer",
    )
    parser.add_argument("--model-path", required=True, help="Model path or HF model ID")
    parser.add_argument("--input-text", required=True, help="Input text to analyze")
    parser.add_argument("--output-dir", required=True, help="Directory for results JSON")
    parser.add_argument("--base-model", default=None, help="Base model for HF loading")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k tokens per position")
    parser.add_argument("--layer-indices", default="", help="Comma-separated layer indices")
