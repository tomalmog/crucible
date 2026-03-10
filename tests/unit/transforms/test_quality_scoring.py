"""Unit tests for configurable quality scoring."""

from __future__ import annotations

import pytest

from core.errors import CrucibleTransformError
from transforms.quality_scoring import score_quality


def test_score_quality_hybrid_returns_scores_for_each_text() -> None:
    """Hybrid model should produce one score per text."""
    scores = score_quality(["clean training text", "another training sample"], "hybrid")

    assert len(scores) == 2


def test_score_quality_perplexity_sets_model_name() -> None:
    """Perplexity model should mark score rows with model name."""
    scores = score_quality(["sample text"], "perplexity")

    assert scores[0].model_name == "perplexity"


def test_score_quality_raises_for_unknown_model() -> None:
    """Unknown model name should raise a transform error."""
    with pytest.raises(CrucibleTransformError):
        score_quality(["sample text"], "unknown-model")

    assert True
