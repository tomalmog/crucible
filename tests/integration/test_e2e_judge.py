"""End-to-end integration tests for the LLM-as-Judge workflow.

Tests cover the LlmJudge SDK methods (evaluate_response, evaluate_model)
with mocked urllib for HTTP calls, API failure fallback behavior,
and the CLI judge subcommand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from cli.main import main
from eval.llm_judge import DEFAULT_CRITERIA, JudgeCriteria, JudgeResult, JudgeScore, LlmJudge


def _make_judge_response(score: float, explanation: str) -> bytes:
    """Build a fake OpenAI-style JSON response body."""
    return json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "score": score,
                    "explanation": explanation,
                }),
            },
        }],
    }).encode()


class _FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _mock_urlopen_factory(score: float = 8.0, explanation: str = "Good"):
    """Return a urlopen replacement that always returns the given score."""
    def _mock_urlopen(request: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(_make_judge_response(score, explanation))
    return _mock_urlopen


# ------------------------------------------------------------------
# SDK tests
# ------------------------------------------------------------------

def test_evaluate_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_response should return one JudgeScore per default criteria."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _mock_urlopen_factory(8.0, "Good"),
    )
    judge = LlmJudge(judge_api_url="http://fake-api/v1/chat/completions")

    scores = judge.evaluate_response("What is AI?", "AI is artificial intelligence.")

    assert len(scores) == len(DEFAULT_CRITERIA)
    for s in scores:
        assert isinstance(s, JudgeScore)
        assert s.score == 8.0
        assert s.explanation == "Good"
    criteria_names = {s.criteria for s in scores}
    assert criteria_names == {c.name for c in DEFAULT_CRITERIA}


def test_evaluate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_model should aggregate scores across multiple prompts."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _mock_urlopen_factory(8.0, "Good"),
    )
    judge = LlmJudge(judge_api_url="http://fake-api/v1/chat/completions")

    result = judge.evaluate_model(
        "model.pt",
        test_prompts=["p1", "p2"],
        responses=["r1", "r2"],
    )

    assert isinstance(result, JudgeResult)
    assert result.average_score == 8.0
    assert result.num_prompts == 2
    assert result.model_path == "model.pt"
    assert len(result.scores) == len(DEFAULT_CRITERIA) * 2


def test_api_failure_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the judge API fails, scores should fall back to 5.0."""
    def _raise_url_error(request: Any, **kwargs: Any) -> None:
        raise URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)
    judge = LlmJudge(judge_api_url="http://unreachable/v1/chat/completions")

    scores = judge.evaluate_response("prompt", "response")

    assert len(scores) == len(DEFAULT_CRITERIA)
    for s in scores:
        assert s.score == 5.0
        assert "failed" in s.explanation.lower() or "default" in s.explanation.lower()


def test_custom_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    """LlmJudge with a single custom criteria should return exactly 1 score."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _mock_urlopen_factory(9.0, "Excellent"),
    )
    custom = (JudgeCriteria("creativity", "How creative is the response"),)
    judge = LlmJudge(
        judge_api_url="http://fake-api/v1/chat/completions",
        criteria=custom,
    )

    scores = judge.evaluate_response("Write a poem", "Roses are red...")

    assert len(scores) == 1
    assert scores[0].criteria == "creativity"
    assert scores[0].score == 9.0


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------

def test_cli_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI judge command should exit 0 and print scores."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _mock_urlopen_factory(7.5, "Decent"),
    )
    prompts_file = tmp_path / "test_prompts.jsonl"
    prompts_file.write_text(
        json.dumps({"prompt": "What is AI?"}) + "\n"
        + json.dumps({"prompt": "Explain transformers."}) + "\n"
    )

    exit_code = main([
        "--data-root", str(tmp_path),
        "judge",
        "--model-path", "m",
        "--judge-api", "http://mock/v1/chat/completions",
        "--test-prompts", str(prompts_file),
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "average_score" in captured
    assert "num_prompts=2" in captured
