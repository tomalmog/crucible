"""Tests for LLM-as-Judge evaluation."""

from __future__ import annotations

from eval.llm_judge import (
    DEFAULT_CRITERIA,
    JudgeCriteria,
    JudgeResult,
    JudgeScore,
    LlmJudge,
)


def test_judge_criteria_defaults() -> None:
    """Default criteria are defined."""
    assert len(DEFAULT_CRITERIA) == 4
    names = [c.name for c in DEFAULT_CRITERIA]
    assert "helpfulness" in names
    assert "accuracy" in names
    assert "safety" in names
    assert "reasoning" in names


def test_judge_score_creation() -> None:
    """JudgeScore stores score and explanation."""
    score = JudgeScore(criteria="helpfulness", score=8.5, explanation="Very helpful")
    assert score.criteria == "helpfulness"
    assert score.score == 8.5
    assert score.explanation == "Very helpful"


def test_judge_result_creation() -> None:
    """JudgeResult aggregates scores."""
    result = JudgeResult(
        model_path="model.pt",
        scores=(JudgeScore("helpfulness", 8.0), JudgeScore("accuracy", 7.0)),
        average_score=7.5,
        num_prompts=5,
    )
    assert result.average_score == 7.5
    assert len(result.scores) == 2


def test_judge_criteria_frozen() -> None:
    """JudgeCriteria is immutable."""
    import pytest
    c = JudgeCriteria("test", "test desc")
    with pytest.raises(AttributeError):
        c.name = "other"  # type: ignore[misc]


def test_llm_judge_evaluate_response() -> None:
    """LlmJudge.evaluate_response returns scores for all criteria."""
    judge = LlmJudge(
        judge_api_url="http://localhost:9999/v1/chat/completions",
        criteria=(JudgeCriteria("helpfulness", "test"),),
    )
    scores = judge.evaluate_response("hello", "hi there")
    assert len(scores) == 1
    assert scores[0].criteria == "helpfulness"
