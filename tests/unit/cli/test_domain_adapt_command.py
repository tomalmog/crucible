"""Unit tests for domain-adapt CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_domain_adapt_command_registers_in_parser() -> None:
    """domain-adapt subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "domain-adapt",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--base-model-path", "/tmp/base.pt",
    ])
    assert args.command == "domain-adapt"
    assert args.dataset == "demo"
    assert args.output_dir == "/tmp/out"
    assert args.base_model_path == "/tmp/base.pt"
    assert args.reference_data_path is None
    assert args.drift_check_interval == 1
    assert args.max_perplexity_increase == 1.5
    assert args.save_best_checkpoint is True


def test_domain_adapt_command_requires_base_model_path() -> None:
    """domain-adapt should fail when --base-model-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "domain-adapt",
            "--dataset", "demo",
            "--output-dir", "/tmp/out",
        ])


def test_domain_adapt_command_requires_dataset() -> None:
    """domain-adapt should fail when --dataset is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "domain-adapt",
            "--output-dir", "/tmp/out",
            "--base-model-path", "/tmp/base.pt",
        ])


def test_domain_adapt_command_optional_args() -> None:
    """domain-adapt should accept optional drift detection args."""
    parser = build_parser()
    args = parser.parse_args([
        "domain-adapt",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--base-model-path", "/tmp/base.pt",
        "--reference-data-path", "/tmp/ref.txt",
        "--drift-check-interval", "2",
        "--max-perplexity-increase", "2.0",
        "--epochs", "5",
        "--learning-rate", "0.001",
    ])
    assert args.reference_data_path == "/tmp/ref.txt"
    assert args.drift_check_interval == 2
    assert args.max_perplexity_increase == 2.0
    assert args.epochs == 5
    assert args.learning_rate == 0.001
