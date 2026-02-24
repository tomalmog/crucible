"""Unit tests for safety gate."""

from __future__ import annotations

import pytest

from core.errors import ForgeSafetyError
from core.safety_types import SafetyEvalConfig
from safety.safety_gate import format_gate_result, run_safety_gate


def test_safety_gate_passes_clean_text() -> None:
    """Safety gate should pass when all texts are clean."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
        toxicity_threshold=0.5,
    )
    texts = ["Hello world", "The sky is blue", "Have a nice day"]
    result = run_safety_gate(config, texts)
    assert result.passed is True
    assert result.report.mean_score < 0.5
    assert result.gate_name == "toxicity-gate"


def test_safety_gate_fails_toxic_text() -> None:
    """Safety gate should fail when texts exceed threshold."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
        toxicity_threshold=0.3,
    )
    texts = [
        "I will kill and murder and attack everyone.",
        "Destroy and hate and violence.",
        "Abuse and torture the threat.",
    ]
    result = run_safety_gate(config, texts)
    assert result.passed is False
    assert result.report.mean_score >= 0.3


def test_safety_gate_raises_on_empty_texts() -> None:
    """Safety gate should raise ForgeSafetyError with no texts."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
    )
    with pytest.raises(ForgeSafetyError, match="No texts provided"):
        run_safety_gate(config, [])


def test_format_gate_result_output() -> None:
    """format_gate_result should return tuple of strings."""
    config = SafetyEvalConfig(
        model_path="/tmp/model.pt",
        eval_data_path="/tmp/data.json",
        output_dir="/tmp/out",
        toxicity_threshold=0.5,
    )
    texts = ["Clean text here"]
    result = run_safety_gate(config, texts)
    lines = format_gate_result(result)
    assert isinstance(lines, tuple)
    assert any("PASSED" in line for line in lines)
