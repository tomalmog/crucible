"""Unit tests for benchmark CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_benchmark_command_registers_in_parser() -> None:
    """Benchmark subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "benchmark",
        "--model-path", "/tmp/model.pt",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
    ])

    assert args.command == "benchmark"
    assert args.model_path == "/tmp/model.pt"
    assert args.dataset == "demo"
    assert args.output_dir == "/tmp/out"
    assert args.max_token_length == 512
    assert args.batch_size == 16
    assert args.no_perplexity is False
    assert args.no_latency is False


def test_benchmark_command_accepts_optional_args() -> None:
    """Benchmark subcommand should accept optional arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "benchmark",
        "--model-path", "/tmp/model.pt",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--max-token-length", "256",
        "--batch-size", "32",
        "--no-perplexity",
        "--no-latency",
    ])

    assert args.max_token_length == 256
    assert args.batch_size == 32
    assert args.no_perplexity is True
    assert args.no_latency is True


def test_benchmark_command_requires_model_path() -> None:
    """Benchmark command should fail when --model-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark",
            "--dataset", "demo",
            "--output-dir", "/tmp/out",
        ])


def test_benchmark_command_requires_dataset() -> None:
    """Benchmark command should fail when --dataset is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark",
            "--model-path", "/tmp/model.pt",
            "--output-dir", "/tmp/out",
        ])


def test_benchmark_command_requires_output_dir() -> None:
    """Benchmark command should fail when --output-dir is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark",
            "--model-path", "/tmp/model.pt",
            "--dataset", "demo",
        ])
