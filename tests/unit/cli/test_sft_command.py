"""Unit tests for SFT CLI command wiring."""

from __future__ import annotations

from cli.main import build_parser


def test_sft_command_registers_in_parser() -> None:
    """SFT subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "sft",
        "--output-dir", "/tmp/out",
        "--sft-data-path", "/tmp/sft.jsonl",
    ])

    assert args.command == "sft"
    assert args.output_dir == "/tmp/out"
    assert args.sft_data_path == "/tmp/sft.jsonl"
    assert args.mask_prompt_tokens is True
    assert args.packing is False


