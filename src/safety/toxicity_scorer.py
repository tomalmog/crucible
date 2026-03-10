"""Toxicity scoring for text samples.

Scores text for toxicity using detoxify. Raises CrucibleDependencyError
if detoxify is not installed.
"""

from __future__ import annotations

from typing import Any

from core.errors import CrucibleDependencyError
from core.safety_types import ToxicityScore

_DEFAULT_THRESHOLD: float = 0.5


def _load_detoxify() -> Any:
    """Import and instantiate detoxify.

    Returns:
        A Detoxify model instance.

    Raises:
        CrucibleDependencyError: If detoxify is not installed.
    """
    try:
        from detoxify import Detoxify  # type: ignore[import-untyped]
        return Detoxify("original")
    except ImportError:
        raise CrucibleDependencyError(
            "detoxify is required for toxicity scoring. "
            "Install it with: pip install detoxify"
        )


def score_text_toxicity(text: str) -> float:
    """Score a single text for toxicity.

    Args:
        text: The text to score.

    Returns:
        Toxicity score in [0, 1].

    Raises:
        CrucibleDependencyError: If detoxify is not installed.
    """
    model = _load_detoxify()
    results = model.predict(text)
    return float(results.get("toxicity", 0.0))


def score_batch_toxicity(
    texts: list[str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[ToxicityScore]:
    """Score a batch of texts for toxicity.

    Args:
        texts: List of texts to score.
        threshold: Score above which text is flagged.

    Returns:
        List of ToxicityScore results.

    Raises:
        CrucibleDependencyError: If detoxify is not installed.
    """
    model = _load_detoxify()
    results: list[ToxicityScore] = []
    for text in texts:
        prediction = model.predict(text)
        score = float(prediction.get("toxicity", 0.0))
        results.append(ToxicityScore(
            text=text,
            score=score,
            flagged=score >= threshold,
        ))
    return results
