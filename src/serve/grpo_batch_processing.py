"""GRPO group sampling and advantage calculation.

This module handles generating groups of responses and computing
group-relative advantages for policy gradient updates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrpoGroupResult:
    """Result of group sampling and advantage calculation.

    Attributes:
        prompt: The input prompt.
        responses: Generated responses in the group.
        rewards: Raw reward scores per response.
        advantages: Normalized advantages per response.
    """

    prompt: str
    responses: list[str]
    rewards: list[float]
    advantages: list[float]


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute advantages normalized within a group.

    Each advantage is (reward - group_mean) / (group_std + eps).
    """
    if not rewards:
        return []
    n = len(rewards)
    mean = sum(rewards) / n
    if n < 2:
        return [0.0]
    variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    std = variance**0.5
    eps = 1e-8
    return [(r - mean) / (std + eps) for r in rewards]


def build_grpo_groups(
    prompts: list[str],
    group_size: int,
) -> list[list[str]]:
    """Replicate each prompt into groups for batch generation.

    Returns a list of groups, each containing group_size copies of a prompt.
    """
    return [[p] * group_size for p in prompts]
