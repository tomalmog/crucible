"""Unit tests for DPO CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_dpo_command_registers_in_parser() -> None:
    """DPO subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "dpo-train",
        "--output-dir", "/tmp/out",
        "--dpo-data-path", "/tmp/dpo.jsonl",
    ])

    assert args.command == "dpo-train"
    assert args.output_dir == "/tmp/out"
    assert args.dpo_data_path == "/tmp/dpo.jsonl"
    assert args.beta == 0.1
    assert args.label_smoothing == 0.0
    assert args.reference_model_path is None


def test_dpo_command_requires_dpo_data_path() -> None:
    """DPO command should fail when --dpo-data-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "dpo-train",
            "--output-dir", "/tmp/out",
        ])


def test_dpo_command_accepts_optional_args() -> None:
    """DPO command should accept optional DPO-specific arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "dpo-train",
        "--output-dir", "/tmp/out",
        "--dpo-data-path", "/tmp/dpo.jsonl",
        "--beta", "0.5",
        "--label-smoothing", "0.1",
        "--reference-model-path", "/tmp/ref_model.pt",
        "--epochs", "5",
        "--learning-rate", "0.001",
    ])

    assert args.beta == 0.5
    assert args.label_smoothing == 0.1
    assert args.reference_model_path == "/tmp/ref_model.pt"
    assert args.epochs == 5
    assert args.learning_rate == 0.001
