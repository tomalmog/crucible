"""Tests for KTO loss function."""

from __future__ import annotations

import pytest


def test_kto_loss_import() -> None:
    """KTO loss module imports correctly."""
    from serve.kto_loss import build_kto_loss_function
    assert callable(build_kto_loss_function)


def test_kto_loss_with_torch() -> None:
    """KTO loss computes with torch tensors."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from serve.kto_loss import build_kto_loss_function
    loss_fn = build_kto_loss_function(torch, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0)
    logits = torch.randn(2, 10, 100)
    targets = torch.randint(0, 100, (2, 10))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
