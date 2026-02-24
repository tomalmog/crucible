"""Unit tests for toxicity scorer."""

from __future__ import annotations

from safety.toxicity_scorer import score_batch_toxicity, score_text_toxicity


def test_clean_text_scores_low() -> None:
    """Clean text should receive a low toxicity score."""
    score = score_text_toxicity("The weather is nice today.")
    assert score < 0.1


def test_toxic_text_scores_high() -> None:
    """Text with toxic keywords should receive a high score."""
    score = score_text_toxicity("I will kill and murder everyone.")
    assert score >= 0.8


def test_batch_scoring_returns_correct_count() -> None:
    """Batch scoring should return one result per input text."""
    texts = ["Hello world", "This is fine", "Nice day"]
    results = score_batch_toxicity(texts)
    assert len(results) == 3


def test_batch_scoring_flags_toxic_text() -> None:
    """Batch scoring should flag toxic text above threshold."""
    texts = [
        "The sun is shining brightly.",
        "I will attack and destroy the target.",
    ]
    results = score_batch_toxicity(texts, threshold=0.2)
    assert not results[0].flagged
    assert results[1].flagged
    assert results[1].score >= 0.2
