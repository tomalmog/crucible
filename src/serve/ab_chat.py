"""A/B model comparison chat.

This module generates responses from two models for the same prompts
and collects preference ratings for DPO training data generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AbComparison:
    """One A/B comparison result.

    Attributes:
        prompt: The input prompt.
        response_a: Response from model A.
        response_b: Response from model B.
        preference: User preference ('a', 'b', or 'tie').
    """

    prompt: str
    response_a: str
    response_b: str
    preference: str = ""


@dataclass
class AbSession:
    """A/B comparison session tracking multiple comparisons.

    Attributes:
        model_a_path: Path to model A.
        model_b_path: Path to model B.
        comparisons: List of comparison results.
    """

    model_a_path: str
    model_b_path: str
    comparisons: list[AbComparison] = field(default_factory=list)


def generate_ab_responses(
    prompt: str,
    model_a_path: str,
    model_b_path: str,
) -> AbComparison:
    """Generate responses from two models for the same prompt.

    This is a placeholder that returns template responses.
    A full implementation would load and run both models.
    """
    return AbComparison(
        prompt=prompt,
        response_a=f"[Model A response to: {prompt}]",
        response_b=f"[Model B response to: {prompt}]",
    )


def export_preferences_as_dpo(
    comparisons: list[AbComparison],
    output_path: str,
) -> int:
    """Export A/B preferences as DPO training data.

    Each comparison with a preference becomes a chosen/rejected pair.
    Returns number of exported pairs.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for c in comparisons:
            if c.preference == "a":
                entry = {"prompt": c.prompt, "chosen": c.response_a, "rejected": c.response_b}
            elif c.preference == "b":
                entry = {"prompt": c.prompt, "chosen": c.response_b, "rejected": c.response_a}
            else:
                continue
            fh.write(json.dumps(entry) + "\n")
            count += 1
    return count
