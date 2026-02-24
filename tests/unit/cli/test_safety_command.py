"""Unit tests for safety CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_safety_eval_command_registers_in_parser() -> None:
    """safety-eval subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "safety-eval",
        "--model-path", "/tmp/model.pt",
        "--eval-data", "/tmp/data.json",
        "--output-dir", "/tmp/out",
    ])
    assert args.command == "safety-eval"
    assert args.model_path == "/tmp/model.pt"
    assert args.eval_data == "/tmp/data.json"
    assert args.output_dir == "/tmp/out"


def test_safety_gate_command_registers_in_parser() -> None:
    """safety-gate subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "safety-gate",
        "--model-path", "/tmp/model.pt",
        "--eval-data", "/tmp/data.json",
        "--output-dir", "/tmp/out",
        "--threshold", "0.3",
    ])
    assert args.command == "safety-gate"
    assert args.model_path == "/tmp/model.pt"
    assert args.eval_data == "/tmp/data.json"
    assert args.output_dir == "/tmp/out"
    assert args.threshold == pytest.approx(0.3)


def test_safety_gate_default_threshold() -> None:
    """safety-gate should default to 0.5 threshold."""
    parser = build_parser()
    args = parser.parse_args([
        "safety-gate",
        "--model-path", "/tmp/model.pt",
        "--eval-data", "/tmp/data.json",
        "--output-dir", "/tmp/out",
    ])
    assert args.threshold == pytest.approx(0.5)
