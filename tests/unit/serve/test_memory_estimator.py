"""Tests for training memory estimator."""

from __future__ import annotations

from serve.memory_estimator import (
    MemoryEstimate,
    estimate_model_memory,
    estimate_training_memory,
)


def test_estimate_model_memory_small() -> None:
    """Small model uses little memory."""
    mem = estimate_model_memory(
        hidden_dim=256, num_layers=2, attention_heads=4, vocab_size=1000,
    )
    assert 0.0 < mem < 0.1


def test_estimate_model_memory_large() -> None:
    """Larger model uses more memory."""
    small = estimate_model_memory(hidden_dim=256, num_layers=2, attention_heads=4)
    large = estimate_model_memory(hidden_dim=4096, num_layers=32, attention_heads=32)
    assert large > small


def test_estimate_training_memory_returns_estimate() -> None:
    """Full estimate returns all fields."""
    est = estimate_training_memory(
        hidden_dim=256, num_layers=2, attention_heads=4,
        batch_size=4, max_token_length=128,
    )
    assert isinstance(est, MemoryEstimate)
    assert est.total_memory_gb > 0
    assert est.model_memory_gb > 0


def test_estimate_fits_in_vram() -> None:
    """Small config fits in large VRAM."""
    est = estimate_training_memory(
        hidden_dim=128, num_layers=1, attention_heads=2,
        batch_size=2, max_token_length=64,
        available_vram_gb=24.0,
    )
    assert est.fits_in_vram is True
