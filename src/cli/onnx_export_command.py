"""CLI command for ONNX model export."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from store.dataset_sdk import CrucibleClient


def add_onnx_export_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the onnx-export subcommand."""
    parser = subparsers.add_parser("onnx-export", help="Export model to ONNX format")
    parser.add_argument("--model-path", required=True, help="Path or ID of the model to export")
    parser.add_argument("--output-dir", required=True, help="Directory for exported ONNX artifacts")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version (default: 17)")


def run_onnx_export_command(client: "CrucibleClient", args: argparse.Namespace) -> int:
    """Execute the onnx-export command."""
    from core.onnx_export_types import OnnxExportOptions
    from serve.export_helpers import resolve_model_path
    from serve.onnx_exporter import run_onnx_export

    resolved = resolve_model_path(args.model_path, client._config.data_root)
    options = OnnxExportOptions(
        model_path=resolved,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
    )
    run_onnx_export(options)
    return 0
