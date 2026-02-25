"""Synthetic data generation.

This module generates instruction-response pairs from seed prompts
using a loaded model, with quality filtering and ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheticExample:
    """One synthetically generated example.

    Attributes:
        prompt: The seed prompt.
        response: Generated response.
        quality_score: Estimated quality (0-1).
    """

    prompt: str
    response: str
    quality_score: float = 0.0


def generate_synthetic_data(
    seed_prompts: list[str],
    count: int = 1000,
    model_path: str | None = None,
) -> list[SyntheticExample]:
    """Generate synthetic instruction-response pairs.

    This is a placeholder that creates template responses.
    A full implementation would load the model and generate.
    """
    examples: list[SyntheticExample] = []
    for i in range(min(count, len(seed_prompts) * 10)):
        prompt = seed_prompts[i % len(seed_prompts)]
        variation = f"Variation {i + 1}: {prompt}"
        response = f"[Generated response for: {variation}]"
        quality = 0.7 + (i % 3) * 0.1
        examples.append(SyntheticExample(
            prompt=variation, response=response,
            quality_score=min(1.0, quality),
        ))
    return examples


def filter_by_quality(
    examples: list[SyntheticExample],
    min_quality: float = 0.5,
) -> list[SyntheticExample]:
    """Filter synthetic examples by quality score."""
    return [e for e in examples if e.quality_score >= min_quality]


def export_synthetic_data(
    examples: list[SyntheticExample],
    output_path: str,
) -> int:
    """Export synthetic examples as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({
                "prompt": ex.prompt,
                "response": ex.response,
                "quality_score": ex.quality_score,
            }) + "\n")
    return len(examples)
