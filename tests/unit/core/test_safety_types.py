"""Unit tests for safety type definitions."""

from __future__ import annotations

import pytest

from core.safety_types import (
    RedTeamResult,
    SafetyEvalConfig,
    SafetyGateResult,
    SafetyReport,
    ToxicityScore,
)


def test_safety_eval_config_is_frozen() -> None:
    """SafetyEvalConfig should be immutable after construction."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
    )
    with pytest.raises(AttributeError):
        config.model_path = "/changed"  # type: ignore[misc]


def test_safety_eval_config_defaults() -> None:
    """SafetyEvalConfig should have correct default values."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
    )
    assert config.toxicity_threshold == 0.5
    assert config.max_samples == 100


def test_safety_report_fields_present() -> None:
    """SafetyReport should expose all expected fields."""
    score = ToxicityScore(text="hello", score=0.1, flagged=False)
    report = SafetyReport(
        total_samples=1,
        flagged_count=0,
        mean_score=0.1,
        max_score=0.1,
        scores=(score,),
    )
    assert report.total_samples == 1
    assert report.flagged_count == 0
    assert report.mean_score == 0.1
    assert report.max_score == 0.1
    assert len(report.scores) == 1
    assert report.scores[0].text == "hello"


def test_safety_gate_result_is_frozen() -> None:
    """SafetyGateResult should be immutable after construction."""
    score = ToxicityScore(text="hi", score=0.05, flagged=False)
    report = SafetyReport(
        total_samples=1,
        flagged_count=0,
        mean_score=0.05,
        max_score=0.05,
        scores=(score,),
    )
    result = SafetyGateResult(
        passed=True,
        report=report,
        threshold=0.5,
        gate_name="test-gate",
    )
    with pytest.raises(AttributeError):
        result.passed = False  # type: ignore[misc]
    assert result.gate_name == "test-gate"


def test_red_team_result_fields() -> None:
    """RedTeamResult should expose all expected fields."""
    result = RedTeamResult(
        suite_name="test-suite",
        total_prompts=10,
        failures=2,
        failure_rate=0.2,
    )
    assert result.suite_name == "test-suite"
    assert result.total_prompts == 10
    assert result.failures == 2
    assert result.failure_rate == pytest.approx(0.2)
