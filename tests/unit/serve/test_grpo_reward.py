"""Tests for GRPO reward function loading and scoring."""

from __future__ import annotations

import pytest

from serve.grpo_reward import default_reward_function, score_responses


def test_default_reward_empty_response() -> None:
    """Empty response gets zero reward."""
    assert default_reward_function("test", "") == 0.0
    assert default_reward_function("test", "   ") == 0.0


def test_default_reward_short_response() -> None:
    """Short response gets proportional reward."""
    score = default_reward_function("test", "hello world")
    assert 0.0 < score < 1.0


def test_default_reward_long_response() -> None:
    """Long response caps at 1.0."""
    long_text = " ".join(["word"] * 100)
    score = default_reward_function("test", long_text)
    assert score == 1.0


def test_score_responses_multiple() -> None:
    """Score multiple responses for a prompt."""
    scores = score_responses(
        default_reward_function,
        "test prompt",
        ["short", "a bit longer response here", ""],
    )
    assert len(scores) == 3
    assert scores[0] > 0.0
    assert scores[1] > scores[0]
    assert scores[2] == 0.0
