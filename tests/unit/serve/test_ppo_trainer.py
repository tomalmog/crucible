"""Unit tests for PPO trainer computations."""

from __future__ import annotations

import torch

from serve.ppo_trainer import compute_advantages, compute_ppo_loss


def test_compute_advantages_normalized() -> None:
    """Advantages should be approximately zero-mean after normalization."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    values = torch.tensor([1.5, 2.5, 2.5, 3.5, 4.5])
    advantages = compute_advantages(
        torch, rewards, values, gamma=1.0, lam=0.95,
    )
    assert advantages.shape == (5,)
    assert torch.isfinite(advantages).all()
    # Mean should be near zero after normalization
    assert abs(advantages.mean().item()) < 0.01


def test_compute_advantages_single_element() -> None:
    """Single-element advantages should not raise errors."""
    rewards = torch.tensor([3.0])
    values = torch.tensor([2.0])
    advantages = compute_advantages(torch, rewards, values, gamma=1.0, lam=0.95)
    assert advantages.shape == (1,)
    assert torch.isfinite(advantages).all()


def test_compute_ppo_loss_no_clipping() -> None:
    """When ratio is within clip range, loss equals unclipped surrogate."""
    log_probs = torch.tensor([-1.0, -2.0, -1.5])
    old_log_probs = torch.tensor([-1.0, -2.0, -1.5])
    advantages = torch.tensor([1.0, -0.5, 0.3])
    loss = compute_ppo_loss(
        torch, log_probs, old_log_probs, advantages, clip_epsilon=0.2,
    )
    assert torch.isfinite(loss)
    # When log_probs == old_log_probs, ratio = 1.0, so loss = -mean(advantages)
    expected = -advantages.mean()
    assert abs(loss.item() - expected.item()) < 0.01


def test_compute_ppo_loss_with_clipping() -> None:
    """Large ratio differences should be clipped."""
    log_probs = torch.tensor([0.0])
    old_log_probs = torch.tensor([-5.0])
    advantages = torch.tensor([1.0])
    loss = compute_ppo_loss(
        torch, log_probs, old_log_probs, advantages, clip_epsilon=0.2,
    )
    assert torch.isfinite(loss)
    # The ratio exp(0 - (-5)) = exp(5) >> 1.2, so clipped ratio = 1.2
    # Loss = -min(exp(5)*1.0, 1.2*1.0) = -1.2
    assert abs(loss.item() - (-1.2)) < 0.01


def test_compute_ppo_loss_negative_advantages() -> None:
    """PPO loss should handle negative advantages correctly."""
    log_probs = torch.tensor([-1.0, -1.5])
    old_log_probs = torch.tensor([-1.0, -1.5])
    advantages = torch.tensor([-1.0, -2.0])
    loss = compute_ppo_loss(
        torch, log_probs, old_log_probs, advantages, clip_epsilon=0.2,
    )
    assert torch.isfinite(loss)
    expected = -advantages.mean()
    assert abs(loss.item() - expected.item()) < 0.01
