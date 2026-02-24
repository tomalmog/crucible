"""Unit tests for deploy CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_deploy_command_registers_in_parser() -> None:
    """Deploy subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "deploy", "package",
        "--model-path", "/tmp/model.onnx",
        "--output-dir", "/tmp/out",
    ])

    assert args.command == "deploy"
    assert args.deploy_action == "package"
    assert args.model_path == "/tmp/model.onnx"
    assert args.output_dir == "/tmp/out"


def test_deploy_quantize_subcommand_parsing() -> None:
    """Deploy quantize subcommand should parse type argument."""
    parser = build_parser()
    args = parser.parse_args([
        "deploy", "quantize",
        "--model-path", "/tmp/model.onnx",
        "--output-dir", "/tmp/out",
        "--type", "static",
    ])

    assert args.deploy_action == "quantize"
    assert args.quant_type == "static"


def test_deploy_checklist_subcommand_parsing() -> None:
    """Deploy checklist subcommand should require model-path and output-dir."""
    parser = build_parser()
    args = parser.parse_args([
        "deploy", "checklist",
        "--model-path", "/tmp/model.onnx",
        "--output-dir", "/tmp/out",
    ])

    assert args.deploy_action == "checklist"
    assert args.model_path == "/tmp/model.onnx"
    assert args.output_dir == "/tmp/out"
