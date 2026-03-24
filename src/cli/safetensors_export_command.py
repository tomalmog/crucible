"""CLI command for SafeTensors model export."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from store.dataset_sdk import CrucibleClient


def add_safetensors_export_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the safetensors-export subcommand."""
    parser = subparsers.add_parser(
        "safetensors-export", help="Export model to SafeTensors format",
    )
    parser.add_argument(
        "--model-path", required=True, help="Path or ID of the model to export",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for exported SafeTensors artifacts",
    )


def run_safetensors_export_command(
    client: "CrucibleClient", args: argparse.Namespace,
) -> int:
    """Execute the safetensors-export command."""
    from core.safetensors_export_types import SafeTensorsExportOptions
    from serve.export_helpers import resolve_model_path
    from serve.safetensors_exporter import run_safetensors_export

    resolved = resolve_model_path(args.model_path, client._config.data_root)
    options = SafeTensorsExportOptions(
        model_path=resolved,
        output_dir=args.output_dir,
    )
    run_safetensors_export(options)
    return 0
