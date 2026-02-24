"""Unit tests for red-team harness."""

from __future__ import annotations

from safety.red_team_harness import (
    DEFAULT_RED_TEAM_PROMPTS,
    format_red_team_report,
    run_red_team_evaluation,
)


def test_default_prompts_exist() -> None:
    """Default red-team prompts should be a non-empty tuple."""
    assert isinstance(DEFAULT_RED_TEAM_PROMPTS, tuple)
    assert len(DEFAULT_RED_TEAM_PROMPTS) >= 5


def test_red_team_evaluation_runs() -> None:
    """Red-team evaluation should run and return a result."""

    def mock_generate(prompt: str) -> str:
        return "I cannot help with that request."

    result = run_red_team_evaluation(mock_generate)
    assert result.suite_name == "default-red-team"
    assert result.total_prompts == len(DEFAULT_RED_TEAM_PROMPTS)
    assert result.failures == 0
    assert result.failure_rate == 0.0


def test_red_team_evaluation_detects_failures() -> None:
    """Red-team evaluation should detect when model fails to refuse."""

    def mock_compliant_generate(prompt: str) -> str:
        return "Sure, here is exactly what you asked for."

    result = run_red_team_evaluation(mock_compliant_generate)
    assert result.failures == result.total_prompts
    assert result.failure_rate == 1.0


def test_format_red_team_report_output() -> None:
    """format_red_team_report should return tuple of strings."""

    def mock_generate(prompt: str) -> str:
        return "I can't do that."

    result = run_red_team_evaluation(mock_generate)
    lines = format_red_team_report(result)
    assert isinstance(lines, tuple)
    assert any("Red Team Report" in line for line in lines)
