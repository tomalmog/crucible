"""Tests for A/B chat comparison."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.chat_types import ChatResult
from serve.ab_chat import (
    AbComparison,
    export_preferences_as_dpo,
    generate_ab_responses,
)


def _mock_run_chat(records, options):
    return ChatResult(response_text=f"Response from {options.model_path}")


def test_generate_ab_responses() -> None:
    """Generate responses returns comparison."""
    with patch("serve.ab_chat.run_chat", side_effect=_mock_run_chat):
        result = generate_ab_responses("hello", "model_a.pt", "model_b.pt")
    assert result.prompt == "hello"
    assert result.response_a
    assert result.response_b


def test_export_dpo_with_preferences(tmp_path: Path) -> None:
    """Export preferences as DPO data."""
    comparisons = [
        AbComparison("p1", "resp_a1", "resp_b1", preference="a"),
        AbComparison("p2", "resp_a2", "resp_b2", preference="b"),
        AbComparison("p3", "resp_a3", "resp_b3", preference="tie"),
    ]
    output = tmp_path / "dpo.jsonl"
    count = export_preferences_as_dpo(comparisons, str(output))
    assert count == 2
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["chosen"] == "resp_a1"
    assert first["rejected"] == "resp_b1"


def test_export_empty_preferences(tmp_path: Path) -> None:
    """No preferences exports nothing."""
    output = tmp_path / "dpo.jsonl"
    count = export_preferences_as_dpo([], str(output))
    assert count == 0
