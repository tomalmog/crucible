"""CLI command for GGUF model export."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from store.dataset_sdk import CrucibleClient


def add_gguf_export_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the gguf-export subcommand."""
    parser = subparsers.add_parser(
        "gguf-export", help="Export model to GGUF format",
    )
    parser.add_argument(
        "--model-path", required=True, help="Path or ID of the model to export",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for exported GGUF artifacts",
    )
    parser.add_argument(
        "--quant-type", default="F16",
        choices=["F32", "F16", "Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M"],
        help="Quantization type (default: F16)",
    )


def run_gguf_export_command(
    client: "CrucibleClient", args: argparse.Namespace,
) -> int:
    """Execute the gguf-export command."""
    from core.gguf_export_types import GgufExportOptions
    from serve.export_helpers import resolve_model_path
    from serve.gguf_exporter import run_gguf_export

    resolved = resolve_model_path(args.model_path, client._config.data_root)
    options = GgufExportOptions(
        model_path=resolved,
        output_dir=args.output_dir,
        quant_type=args.quant_type,
    )
    run_gguf_export(options)
    return 0
