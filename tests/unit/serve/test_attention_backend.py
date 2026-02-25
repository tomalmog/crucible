"""Tests for attention backend detection and selection."""

from __future__ import annotations

from serve.attention_backend import (
    detect_available_backends,
    select_attention_backend,
)


def test_detect_backends_returns_list() -> None:
    """Backend detection returns a list of backends."""
    backends = detect_available_backends()
    assert len(backends) >= 1
    names = [b.name for b in backends]
    assert "vanilla" in names


def test_vanilla_always_available() -> None:
    """Vanilla attention is always available."""
    backends = detect_available_backends()
    vanilla = [b for b in backends if b.name == "vanilla"][0]
    assert vanilla.available is True


def test_select_auto_returns_valid() -> None:
    """Auto selection returns a valid backend."""
    result = select_attention_backend("auto")
    assert result in ("flash", "sdpa", "vanilla")


def test_select_vanilla_always_works() -> None:
    """Requesting vanilla always succeeds."""
    assert select_attention_backend("vanilla") == "vanilla"
