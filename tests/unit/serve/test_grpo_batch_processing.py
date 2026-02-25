"""Tests for GRPO batch processing utilities."""

from __future__ import annotations

from serve.grpo_batch_processing import build_grpo_groups, compute_group_advantages


def test_compute_advantages_normalized() -> None:
    """Advantages are normalized within a group."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = compute_group_advantages(rewards)
    assert len(advantages) == 4
    assert abs(sum(advantages)) < 1e-6


def test_compute_advantages_empty() -> None:
    """Empty rewards returns empty advantages."""
    assert compute_group_advantages([]) == []


def test_compute_advantages_single() -> None:
    """Single reward returns zero advantage."""
    assert compute_group_advantages([5.0]) == [0.0]


def test_compute_advantages_equal_rewards() -> None:
    """Equal rewards produce near-zero advantages."""
    advantages = compute_group_advantages([3.0, 3.0, 3.0])
    for a in advantages:
        assert abs(a) < 1e-6


def test_build_groups() -> None:
    """Build groups replicates prompts correctly."""
    prompts = ["p1", "p2"]
    groups = build_grpo_groups(prompts, group_size=3)
    assert len(groups) == 2
    assert groups[0] == ["p1", "p1", "p1"]
    assert groups[1] == ["p2", "p2", "p2"]
