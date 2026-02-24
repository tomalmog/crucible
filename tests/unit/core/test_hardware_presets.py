"""Unit tests for hardware preset configurations."""

from __future__ import annotations

from core.hardware_presets import (
    HardwarePreset,
    list_preset_names,
    lookup_hardware_preset,
)


def test_lookup_known_preset_h100() -> None:
    """H100 preset should return bf16 with large batch size."""
    preset = lookup_hardware_preset("h100")
    assert preset.profile_name == "h100"
    assert preset.batch_size == 64
    assert preset.precision_mode == "bf16"
    assert preset.gradient_checkpointing is False


def test_lookup_known_preset_a100() -> None:
    """A100 preset should return bf16."""
    preset = lookup_hardware_preset("a100")
    assert preset.profile_name == "a100"
    assert preset.precision_mode == "bf16"


def test_lookup_known_preset_l40() -> None:
    """L40 preset should enable gradient checkpointing."""
    preset = lookup_hardware_preset("l40")
    assert preset.profile_name == "l40"
    assert preset.gradient_checkpointing is True
    assert preset.gradient_accumulation_steps == 2


def test_lookup_known_preset_cpu() -> None:
    """CPU preset should return fp32 with small batch."""
    preset = lookup_hardware_preset("cpu")
    assert preset.profile_name == "cpu"
    assert preset.precision_mode == "fp32"
    assert preset.batch_size == 4


def test_lookup_unknown_profile_falls_back_to_cpu() -> None:
    """Unknown profile names should fall back to CPU preset."""
    preset = lookup_hardware_preset("unknown_gpu_xyz")
    assert preset.profile_name == "cpu"


def test_list_preset_names_returns_sorted_tuple() -> None:
    """list_preset_names should return all known profiles sorted."""
    names = list_preset_names()
    assert isinstance(names, tuple)
    assert len(names) >= 6
    assert names == tuple(sorted(names))
    assert "h100" in names
    assert "cpu" in names


def test_preset_is_frozen() -> None:
    """HardwarePreset should be immutable."""
    preset = lookup_hardware_preset("h100")
    assert isinstance(preset, HardwarePreset)
    try:
        preset.batch_size = 999  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass
