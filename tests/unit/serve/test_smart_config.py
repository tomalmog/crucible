"""Tests for smart hardware-aware configuration."""

from __future__ import annotations

from serve.smart_config import suggest_training_config


def test_suggest_for_rtx4090() -> None:
    """Suggestion for RTX 4090 uses bf16."""
    suggestion = suggest_training_config(
        model_size_billions=7.0, training_method="sft",
        gpu_name="rtx4090",
    )
    assert suggestion.precision_mode == "bf16"
    assert suggestion.batch_size >= 1


def test_suggest_for_apple_silicon() -> None:
    """Apple Silicon suggestion uses fp32."""
    suggestion = suggest_training_config(
        model_size_billions=1.0, training_method="sft",
        gpu_name="m1",
    )
    assert suggestion.precision_mode == "fp32"


def test_suggest_large_model_uses_qlora() -> None:
    """Large model on small GPU recommends QLoRA."""
    suggestion = suggest_training_config(
        model_size_billions=13.0, training_method="sft",
        gpu_name="rtx4090",
    )
    assert suggestion.use_qlora is True


def test_suggest_small_model_no_qlora() -> None:
    """Small model on large GPU doesn't need QLoRA."""
    suggestion = suggest_training_config(
        model_size_billions=0.1, training_method="sft",
        gpu_name="h100",
    )
    assert suggestion.use_qlora is False
