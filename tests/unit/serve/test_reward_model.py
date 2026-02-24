"""Unit tests for reward model scoring and loading."""

from __future__ import annotations

import torch

from serve.reward_model import (
    build_reward_head,
    compute_reward_scores,
    create_reference_policy,
)


def test_build_reward_head_produces_scalar() -> None:
    """Reward head should map hidden_dim to scalar output."""
    head = build_reward_head(torch, hidden_dim=64)
    input_tensor = torch.randn(4, 64)
    output = head(input_tensor)
    assert output.shape == (4, 1)


def test_compute_reward_scores_returns_batch_scores() -> None:
    """compute_reward_scores should return one score per batch item."""
    model = torch.nn.Linear(16, 1)
    input_ids = torch.randn(3, 16)
    # Use a simple model that produces scalar output
    scores = compute_reward_scores(
        torch, model, input_ids, torch.device("cpu"),
    )
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_create_reference_policy_is_frozen() -> None:
    """Reference policy should have no trainable parameters."""
    model = torch.nn.Linear(8, 4)
    ref = create_reference_policy(torch, model)
    for param in ref.parameters():
        assert not param.requires_grad


def test_create_reference_policy_preserves_weights() -> None:
    """Reference policy should have same weights as original model."""
    model = torch.nn.Linear(8, 4)
    ref = create_reference_policy(torch, model)
    for orig_p, ref_p in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(orig_p, ref_p)
