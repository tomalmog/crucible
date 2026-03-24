"""CLI command for SAE analysis."""

from __future__ import annotations

import argparse
from typing import Any


def add_sae_analyze_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("sae-analyze", help="Analyze text through a trained sparse autoencoder")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sae-path", required=True, help="Path to trained SAE .pt file")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--top-k-features", type=int, default=10)


def run_sae_analyze_command(client: Any, args: Any) -> int:
    from core.sae_types import SaeAnalyzeOptions
    from serve.sae_analyze_runner import run_sae_analyze

    options = SaeAnalyzeOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        sae_path=args.sae_path,
        input_text=args.input_text,
        base_model=getattr(args, "base_model", None),
        top_k_features=args.top_k_features,
    )
    run_sae_analyze(options)
    return 0
