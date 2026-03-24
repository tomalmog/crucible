"""CLI command for HuggingFace model export."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from store.dataset_sdk import CrucibleClient


def add_hf_export_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the hf-export subcommand."""
    parser = subparsers.add_parser(
        "hf-export", help="Export model to HuggingFace-compatible directory",
    )
    parser.add_argument(
        "--model-path", required=True, help="Path or ID of the model to export",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for exported HuggingFace artifacts",
    )


def run_hf_export_command(
    client: "CrucibleClient", args: argparse.Namespace,
) -> int:
    """Execute the hf-export command."""
    from core.hf_export_types import HfExportOptions
    from serve.export_helpers import resolve_model_path
    from serve.hf_exporter import run_hf_export

    resolved = resolve_model_path(args.model_path, client._config.data_root)
    options = HfExportOptions(
        model_path=resolved,
        output_dir=args.output_dir,
    )
    run_hf_export(options)
    return 0
