"""Unit tests for hardware auto-config."""

from __future__ import annotations

from core.hardware_presets import HardwarePreset
from core.types import TrainingOptions
from serve.hardware_auto_config import _apply_preset_to_options


def _make_default_options() -> TrainingOptions:
    return TrainingOptions(dataset_name="test", output_dir="/tmp/out")


def test_apply_preset_overrides_batch_size() -> None:
    """Preset batch_size should override options default."""
    options = _make_default_options()
    preset = HardwarePreset(
        profile_name="test",
        batch_size=48,
        precision_mode="bf16",
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
    )
    result = _apply_preset_to_options(options, preset)
    assert result.batch_size == 48


def test_apply_preset_overrides_precision_mode() -> None:
    """Preset precision_mode should be applied."""
    options = _make_default_options()
    preset = HardwarePreset(
        profile_name="test",
        batch_size=16,
        precision_mode="bf16",
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
    )
    result = _apply_preset_to_options(options, preset)
    assert result.precision_mode == "bf16"


def test_apply_preset_overrides_gradient_checkpointing() -> None:
    """Preset gradient_checkpointing should be applied."""
    options = _make_default_options()
    preset = HardwarePreset(
        profile_name="test",
        batch_size=16,
        precision_mode="fp32",
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
    )
    result = _apply_preset_to_options(options, preset)
    assert result.gradient_checkpointing is True


def test_apply_preset_overrides_accumulation_steps() -> None:
    """Preset gradient_accumulation_steps should be applied."""
    options = _make_default_options()
    preset = HardwarePreset(
        profile_name="test",
        batch_size=16,
        precision_mode="fp32",
        gradient_checkpointing=False,
        gradient_accumulation_steps=4,
    )
    result = _apply_preset_to_options(options, preset)
    assert result.gradient_accumulation_steps == 4


def test_apply_preset_preserves_non_overridden_fields() -> None:
    """Fields not in the preset should remain unchanged."""
    options = TrainingOptions(
        dataset_name="ds",
        output_dir="/out",
        epochs=10,
        learning_rate=0.001,
    )
    preset = HardwarePreset(
        profile_name="test",
        batch_size=32,
        precision_mode="fp16",
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
    )
    result = _apply_preset_to_options(options, preset)
    assert result.epochs == 10
    assert result.learning_rate == 0.001
    assert result.dataset_name == "ds"
