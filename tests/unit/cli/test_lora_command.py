"""Unit tests for LoRA CLI command registration."""

from __future__ import annotations

import argparse

from cli.lora_command import add_lora_merge_command, add_lora_train_command


def test_lora_train_command_registers_in_parser() -> None:
    """lora-train subcommand should be registered with required arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_lora_train_command(subparsers)

    args = parser.parse_args([
        "lora-train",
        "--dataset", "test",
        "--output-dir", "/tmp/out",
        "--sft-data-path", "/tmp/data.jsonl",
        "--base-model-path", "/tmp/model.pt",
    ])
    assert args.dataset == "test"
    assert args.sft_data_path == "/tmp/data.jsonl"
    assert args.base_model_path == "/tmp/model.pt"
    assert args.lora_rank == 8


def test_lora_merge_command_registers_in_parser() -> None:
    """lora-merge subcommand should be registered with required arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_lora_merge_command(subparsers)

    args = parser.parse_args([
        "lora-merge",
        "--base-model-path", "/tmp/model.pt",
        "--adapter-path", "/tmp/adapter.pt",
        "--output-path", "/tmp/merged.pt",
    ])
    assert args.base_model_path == "/tmp/model.pt"
    assert args.adapter_path == "/tmp/adapter.pt"
    assert args.output_path == "/tmp/merged.pt"
