"""End-to-end integration tests for the suggest training config workflow.

Tests cover the smart_config suggestion engine, GPU profile listing,
and CLI entry points for the suggest command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.main import main
from serve.gpu_profiles import list_gpu_profiles
from serve.smart_config import suggest_training_config


def test_small_model_rtx4090() -> None:
    """A 2B model on RTX 4090 should fit without QLoRA and use bf16 or fp16."""
    suggestion = suggest_training_config(2.0, "sft", "rtx4090")

    assert suggestion.batch_size > 0
    assert suggestion.precision_mode in ("bf16", "fp16")
    assert suggestion.use_qlora is False


def test_large_model_triggers_qlora() -> None:
    """A 70B model on RTX 4090 should trigger QLoRA due to VRAM limits."""
    suggestion = suggest_training_config(70.0, "sft", "rtx4090")

    assert suggestion.use_qlora is True


def test_apple_silicon() -> None:
    """Apple Silicon GPUs should use fp32 precision mode."""
    suggestion = suggest_training_config(7.0, "sft", "m3_max")

    assert suggestion.precision_mode == "fp32"


def test_preference_method() -> None:
    """DPO training should produce a note about reference model or non-empty notes."""
    suggestion = suggest_training_config(7.0, "dpo-train", "rtx4090")

    notes_text = " ".join(suggestion.notes).lower()
    assert "reference model" in notes_text or len(suggestion.notes) > 0


def test_cli_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI suggest command should exit 0 and print batch_size."""
    exit_code = main(["--data-root", str(tmp_path), "suggest", "--model-size", "7"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "batch_size=" in captured


def test_cli_list_gpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI suggest --list-gpus should exit 0 and include rtx4090 profile."""
    exit_code = main(["--data-root", str(tmp_path), "suggest", "--list-gpus"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "rtx4090" in captured.lower() or "RTX 4090" in captured
