"""Configurable quality scoring models.

This module provides model selection for text quality scoring.
It currently supports a stronger hybrid scorer and baseline perplexity.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import DEFAULT_QUALITY_MODEL, SUPPORTED_QUALITY_MODELS
from core.errors import CrucibleTransformError
from transforms.perplexity_quality import score_texts_with_perplexity


@dataclass(frozen=True)
class QualityScore:
    """Quality scoring result for one document.

    Attributes:
        perplexity: Model perplexity signal.
        quality_score: Final normalized quality score.
        model_name: Quality model identifier.
    """

    perplexity: float
    quality_score: float
    model_name: str


def score_quality(texts: list[str], model_name: str = DEFAULT_QUALITY_MODEL) -> list[QualityScore]:
    """Score text quality with the selected model.

    Args:
        texts: Input text documents.
        model_name: Quality model identifier.

    Returns:
        Ordered quality score results.

    Raises:
        CrucibleTransformError: If model name is unsupported.
    """
    normalized_model = model_name.lower().strip()
    if normalized_model not in SUPPORTED_QUALITY_MODELS:
        supported = ", ".join(SUPPORTED_QUALITY_MODELS)
        raise CrucibleTransformError(
            f"Unsupported quality model '{model_name}'. "
            f"Choose one of: {supported}."
        )
    perplexity_pairs = score_texts_with_perplexity(texts)
    if normalized_model == "perplexity":
        return _as_perplexity_scores(perplexity_pairs)
    return _build_hybrid_scores(texts, perplexity_pairs)


def _as_perplexity_scores(perplexity_pairs: list[tuple[float, float]]) -> list[QualityScore]:
    """Convert perplexity tuples into QualityScore objects."""
    return [
        QualityScore(perplexity=perplexity, quality_score=score, model_name="perplexity")
        for perplexity, score in perplexity_pairs
    ]


def _build_hybrid_scores(
    texts: list[str],
    perplexity_pairs: list[tuple[float, float]],
) -> list[QualityScore]:
    """Build hybrid scores from perplexity and heuristic signals."""
    hybrid_scores: list[QualityScore] = []
    for text, (perplexity, perplexity_score) in zip(texts, perplexity_pairs):
        heuristic_score = _compute_heuristic_score(text)
        combined_score = round((0.65 * perplexity_score) + (0.35 * heuristic_score), 6)
        hybrid_scores.append(
            QualityScore(
                perplexity=perplexity,
                quality_score=combined_score,
                model_name="hybrid",
            )
        )
    return hybrid_scores


def _compute_heuristic_score(text: str) -> float:
    """Compute heuristic quality score from document features."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count
    alpha_ratio = _count_alpha_characters(text) / max(len(text), 1)
    repeat_penalty = _compute_repeat_penalty(text)
    length_score = _compute_length_score(token_count)
    diversity_score = min(unique_ratio / 0.55, 1.0)
    alpha_score = min(alpha_ratio / 0.75, 1.0)
    score = (0.45 * length_score) + (0.35 * diversity_score) + (0.20 * alpha_score)
    return max(0.0, min(1.0, round(score - repeat_penalty, 6)))


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    cleaned = "".join(character if character.isalnum() else " " for character in text)
    return [token for token in cleaned.lower().split() if token]


def _count_alpha_characters(text: str) -> int:
    """Count alphabetic characters in text."""
    return sum(1 for character in text if character.isalpha())


def _compute_repeat_penalty(text: str) -> float:
    """Compute repetition penalty for long repeated character runs."""
    repeated_runs = 0
    last_character = ""
    run_length = 0
    for character in text:
        if character == last_character:
            run_length += 1
        else:
            if run_length >= 6:
                repeated_runs += 1
            last_character = character
            run_length = 1
    if run_length >= 6:
        repeated_runs += 1
    return min(repeated_runs * 0.05, 0.35)


def _compute_length_score(token_count: int) -> float:
    """Compute smooth score for practical document lengths."""
    if token_count < 20:
        return token_count / 20
    if token_count <= 450:
        return 1.0
    overflow = token_count - 450
    return max(0.1, 1 - (overflow / 900))


def supported_quality_models() -> tuple[str, ...]:
    """Return quality model identifiers for CLI/API usage."""
    return SUPPORTED_QUALITY_MODELS
