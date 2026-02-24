"""Unit tests for sweep CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_sweep_command_registers_in_parser() -> None:
    """Sweep subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "sweep",
        "--dataset", "demo",
        "--output-dir", "/tmp/sweep-out",
        "--config-file", "/tmp/sweep.yaml",
    ])

    assert args.command == "sweep"
    assert args.dataset == "demo"
    assert args.output_dir == "/tmp/sweep-out"
    assert args.config_file == "/tmp/sweep.yaml"
    assert args.strategy == "grid"
    assert args.max_trials == 10
    assert args.metric == "validation_loss"
    assert args.maximize is False


def test_sweep_command_accepts_optional_args() -> None:
    """Sweep subcommand should accept all optional arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "sweep",
        "--dataset", "demo",
        "--output-dir", "/tmp/sweep-out",
        "--config-file", "/tmp/sweep.yaml",
        "--strategy", "random",
        "--max-trials", "20",
        "--metric", "train_loss",
        "--maximize",
    ])

    assert args.strategy == "random"
    assert args.max_trials == 20
    assert args.metric == "train_loss"
    assert args.maximize is True


def test_sweep_command_requires_dataset() -> None:
    """Sweep command should fail when --dataset is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sweep",
            "--output-dir", "/tmp/sweep-out",
            "--config-file", "/tmp/sweep.yaml",
        ])


def test_sweep_command_requires_output_dir() -> None:
    """Sweep command should fail when --output-dir is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sweep",
            "--dataset", "demo",
            "--config-file", "/tmp/sweep.yaml",
        ])


def test_sweep_command_requires_config_file() -> None:
    """Sweep command should fail when --config-file is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sweep",
            "--dataset", "demo",
            "--output-dir", "/tmp/sweep-out",
        ])


def test_sweep_command_rejects_invalid_strategy() -> None:
    """Sweep command should reject unsupported strategy values."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sweep",
            "--dataset", "demo",
            "--output-dir", "/tmp/sweep-out",
            "--config-file", "/tmp/sweep.yaml",
            "--strategy", "bayesian",
        ])
