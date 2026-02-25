"""Tests for ORPO loss function."""

from __future__ import annotations

import pytest


def test_orpo_loss_import() -> None:
    """ORPO loss module imports correctly."""
    from serve.orpo_loss import build_orpo_loss_function
    assert callable(build_orpo_loss_function)


def test_orpo_loss_with_torch() -> None:
    """ORPO loss computes with torch tensors."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from serve.orpo_loss import build_orpo_loss_function
    loss_fn = build_orpo_loss_function(torch, lambda_orpo=1.0, beta=0.1)
    logits = torch.randn(2, 10, 100)
    targets = torch.randint(0, 100, (2, 10))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
