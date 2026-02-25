"""Tests for GPU profiles."""

from __future__ import annotations

from serve.gpu_profiles import get_gpu_profile, list_gpu_profiles


def test_list_profiles_not_empty() -> None:
    """Profile list has entries."""
    profiles = list_gpu_profiles()
    assert len(profiles) >= 5


def test_get_rtx4090() -> None:
    """RTX 4090 profile is available."""
    profile = get_gpu_profile("rtx4090")
    assert profile is not None
    assert profile.vram_gb == 24.0
    assert profile.supports_flash_attention is True


def test_get_h100() -> None:
    """H100 profile is available."""
    profile = get_gpu_profile("h100")
    assert profile is not None
    assert profile.vram_gb == 80.0


def test_get_unknown_returns_none() -> None:
    """Unknown GPU returns None."""
    assert get_gpu_profile("nonexistent_gpu_xyz") is None


def test_apple_silicon_profile() -> None:
    """Apple M-series profiles exist."""
    profile = get_gpu_profile("m1")
    assert profile is not None
    assert profile.supports_flash_attention is False
