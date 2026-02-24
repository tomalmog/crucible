"""Hardware-specific training preset configurations.

This module defines frozen preset profiles for known GPU families.
Each preset recommends training defaults calibrated for the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import PrecisionMode


@dataclass(frozen=True)
class HardwarePreset:
    """Recommended training defaults for a hardware profile."""

    profile_name: str
    batch_size: int
    precision_mode: PrecisionMode
    gradient_checkpointing: bool
    gradient_accumulation_steps: int


_H100_PRESET = HardwarePreset(
    profile_name="h100",
    batch_size=64,
    precision_mode="bf16",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
)

_A100_PRESET = HardwarePreset(
    profile_name="a100",
    batch_size=48,
    precision_mode="bf16",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
)

_L40_PRESET = HardwarePreset(
    profile_name="l40",
    batch_size=24,
    precision_mode="fp16",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
)

_GENERIC_CUDA_PRESET = HardwarePreset(
    profile_name="generic_cuda",
    batch_size=16,
    precision_mode="fp16",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
)

_MPS_PRESET = HardwarePreset(
    profile_name="apple_mps",
    batch_size=8,
    precision_mode="fp32",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
)

_CPU_PRESET = HardwarePreset(
    profile_name="cpu",
    batch_size=4,
    precision_mode="fp32",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
)

_PRESET_REGISTRY: dict[str, HardwarePreset] = {
    "h100": _H100_PRESET,
    "a100": _A100_PRESET,
    "l40": _L40_PRESET,
    "generic_cuda": _GENERIC_CUDA_PRESET,
    "apple_mps": _MPS_PRESET,
    "cpu": _CPU_PRESET,
}


def lookup_hardware_preset(profile_name: str) -> HardwarePreset:
    """Look up a hardware preset by profile name, falling back to CPU."""
    return _PRESET_REGISTRY.get(profile_name, _CPU_PRESET)


def list_preset_names() -> tuple[str, ...]:
    """Return all known preset profile names."""
    return tuple(sorted(_PRESET_REGISTRY.keys()))
