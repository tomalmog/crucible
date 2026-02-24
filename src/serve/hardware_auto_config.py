"""Hardware-aware automatic training configuration.

This module applies hardware profile-based defaults to training options
when the user enables auto-config mode.
"""

from __future__ import annotations

from dataclasses import replace

from core.hardware_presets import HardwarePreset, lookup_hardware_preset
from core.types import TrainingOptions
from serve.hardware_profile import HardwareProfile, detect_hardware_profile


def apply_hardware_auto_config(options: TrainingOptions) -> TrainingOptions:
    """Return options with hardware-recommended defaults applied.

    Detects the local hardware profile, looks up the matching preset,
    and overrides batch_size, precision_mode, gradient_checkpointing,
    and gradient_accumulation_steps with recommended values.
    """
    profile = detect_hardware_profile()
    preset = lookup_hardware_preset(profile.suggested_profile)
    return _apply_preset_to_options(options, preset)


def apply_preset_to_options(
    options: TrainingOptions, profile: HardwareProfile,
) -> TrainingOptions:
    """Apply hardware preset from a pre-detected profile."""
    preset = lookup_hardware_preset(profile.suggested_profile)
    return _apply_preset_to_options(options, preset)


def _apply_preset_to_options(
    options: TrainingOptions, preset: HardwarePreset,
) -> TrainingOptions:
    """Merge preset defaults into training options."""
    return replace(
        options,
        batch_size=preset.batch_size,
        precision_mode=preset.precision_mode,
        gradient_checkpointing=preset.gradient_checkpointing,
        gradient_accumulation_steps=preset.gradient_accumulation_steps,
    )
