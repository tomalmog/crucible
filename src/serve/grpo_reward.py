"""GRPO reward function loading and scoring.

This module loads user-defined reward functions and scores
generated responses for group-relative advantage calculation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from core.errors import CrucibleGrpoError


class RewardFunction(Protocol):
    """Protocol for user-defined reward functions."""

    def __call__(self, prompt: str, response: str) -> float: ...


def load_reward_function(reward_path: str) -> RewardFunction:
    """Load a reward function from a Python module file.

    The module must define a ``score(prompt, response) -> float`` function.
    """
    path = Path(reward_path)
    if not path.exists():
        raise CrucibleGrpoError(f"Reward function file not found: {reward_path}")
    spec = importlib.util.spec_from_file_location("grpo_reward_module", path)
    if spec is None or spec.loader is None:
        raise CrucibleGrpoError(f"Cannot load reward module from: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["grpo_reward_module"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "score", None)
    if fn is None or not callable(fn):
        raise CrucibleGrpoError(
            f"Reward module {reward_path} must define a callable 'score(prompt, response) -> float'."
        )
    return fn


def default_reward_function(prompt: str, response: str) -> float:
    """Built-in length-based reward function used when no custom reward is provided."""
    if not response.strip():
        return 0.0
    word_count = len(response.split())
    length_score = min(word_count / 50.0, 1.0)
    return length_score


def score_responses(
    reward_fn: RewardFunction,
    prompt: str,
    responses: list[str],
) -> list[float]:
    """Score a group of responses for a single prompt."""
    return [reward_fn(prompt, r) for r in responses]
