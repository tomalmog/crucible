"""Tests for dataset curator."""

from __future__ import annotations

from serve.dataset_curator import (
    DatasetDistribution,
    QualityScore,
    compute_distributions,
    score_examples,
)


def test_score_normal_content() -> None:
    """Normal content gets high score."""
    records = [{"id": "1", "text": "This is a normal piece of text with good content and proper formatting."}]
    scores = score_examples(records)
    assert len(scores) == 1
    assert scores[0].score > 0.8
    assert not scores[0].issues


def test_score_short_content() -> None:
    """Short content is flagged."""
    records = [{"id": "1", "text": "Hi"}]
    scores = score_examples(records)
    assert scores[0].score < 0.8
    assert "too_short" in scores[0].issues


def test_score_empty_content() -> None:
    """Empty content gets zero score."""
    records = [{"id": "1", "text": ""}]
    scores = score_examples(records)
    assert scores[0].score == 0.0
    assert "empty_content" in scores[0].issues


def test_score_repetitive_content() -> None:
    """Repetitive content is flagged."""
    records = [{"id": "1", "text": "word " * 100}]
    scores = score_examples(records)
    assert "highly_repetitive" in scores[0].issues


def test_compute_distributions() -> None:
    """Distributions are computed correctly."""
    records = [
        {"text": " ".join(["word"] * 30)},
        {"text": " ".join(["word"] * 75)},
        {"text": " ".join(["word"] * 150)},
    ]
    dist = compute_distributions(records)
    assert dist.total_records == 3
    assert dist.min_token_length == 30
    assert dist.max_token_length == 150
    assert dist.avg_token_length > 0


def test_compute_distributions_empty() -> None:
    """Empty dataset returns zero distributions."""
    dist = compute_distributions([])
    assert dist.total_records == 0
    assert dist.avg_token_length == 0.0
