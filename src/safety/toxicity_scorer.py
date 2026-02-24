"""Toxicity scoring for text samples.

Scores text for toxicity using detoxify when available,
falling back to a simple keyword-based heuristic otherwise.
"""

from __future__ import annotations

from typing import Any

from core.safety_types import ToxicityScore

_TOXIC_KEYWORDS: tuple[str, ...] = (
    "kill", "murder", "attack", "destroy", "hate",
    "violence", "abuse", "torture", "threat", "harm",
)

_DEFAULT_THRESHOLD: float = 0.5


def _try_load_detoxify() -> Any | None:
    """Try to import and instantiate detoxify.

    Returns:
        A Detoxify model instance, or None if unavailable.
    """
    try:
        from detoxify import Detoxify  # type: ignore[import-untyped]
        return Detoxify("original")
    except (ImportError, Exception):
        return None


def _keyword_score(text: str) -> float:
    """Score text using keyword heuristic.

    Args:
        text: Input text to score.

    Returns:
        Score between 0.0 and 1.0.
    """
    lower_text = text.lower()
    matches = sum(1 for kw in _TOXIC_KEYWORDS if kw in lower_text)
    if matches == 0:
        return 0.0
    return min(0.8 + (matches - 1) * 0.05, 1.0)


def score_text_toxicity(text: str) -> float:
    """Score a single text for toxicity.

    Uses detoxify if available, otherwise falls back to
    a keyword-based heuristic.

    Args:
        text: The text to score.

    Returns:
        Toxicity score in [0, 1].
    """
    model = _try_load_detoxify()
    if model is not None:
        results = model.predict(text)
        return float(results.get("toxicity", 0.0))
    return _keyword_score(text)


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
    """
    model = _try_load_detoxify()
    results: list[ToxicityScore] = []
    for text in texts:
        if model is not None:
            prediction = model.predict(text)
            score = float(prediction.get("toxicity", 0.0))
        else:
            score = _keyword_score(text)
        results.append(ToxicityScore(
            text=text,
            score=score,
            flagged=score >= threshold,
        ))
    return results
