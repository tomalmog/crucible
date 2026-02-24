"""Unit tests for SFT CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser, main
from core.types import TrainingRunResult
from store.dataset_sdk import ForgeClient


def test_sft_command_registers_in_parser() -> None:
    """SFT subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "sft",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--sft-data-path", "/tmp/sft.jsonl",
    ])

    assert args.command == "sft"
    assert args.dataset == "demo"
    assert args.output_dir == "/tmp/out"
    assert args.sft_data_path == "/tmp/sft.jsonl"
    assert args.mask_prompt_tokens is True
    assert args.packing is False


def test_sft_command_requires_sft_data_path() -> None:
    """SFT command should fail when --sft-data-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sft",
            "--dataset", "demo",
            "--output-dir", "/tmp/out",
        ])
