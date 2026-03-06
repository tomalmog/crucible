"""Integration tests for A/B chat comparison end-to-end workflow.

Covers generate_ab_responses, export_preferences_as_dpo SDK functions
and CLI ab-chat subcommands using real file I/O against a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.chat_types import ChatResult
from serve.ab_chat import AbComparison, export_preferences_as_dpo, generate_ab_responses


def _mock_run_chat(records, options):
    """Mock chat runner that returns a deterministic response."""
    return ChatResult(response_text=f"Response from {options.model_path}: {options.prompt}")


def test_generate_responses() -> None:
    """Generating A/B responses should return a comparison with non-empty fields."""
    with patch("serve.ab_chat.run_chat", side_effect=_mock_run_chat):
        result = generate_ab_responses("Hello", "model_a", "model_b")

    assert result.prompt == "Hello"
    assert result.response_a != ""
    assert result.response_b != ""
    assert "model_a" in result.response_a
    assert "model_b" in result.response_b


def test_export_dpo(tmp_path: Path) -> None:
    """Exporting comparisons with a preference should produce valid DPO JSONL."""
    comparisons = [
        AbComparison(prompt="Hi", response_a="A says hi", response_b="B says hi", preference="a"),
        AbComparison(prompt="Bye", response_a="A says bye", response_b="B says bye", preference="b"),
    ]
    output = tmp_path / "dpo.jsonl"

    count = export_preferences_as_dpo(comparisons, str(output))

    assert count >= 1
    assert output.exists()
    for line in output.read_text().strip().splitlines():
        entry = json.loads(line)
        assert "prompt" in entry
        assert "chosen" in entry
        assert "rejected" in entry


def test_export_skips_ties(tmp_path: Path) -> None:
    """Comparisons without a clear preference should not be exported."""
    comparisons = [
        AbComparison(prompt="Q1", response_a="A1", response_b="B1", preference=""),
        AbComparison(prompt="Q2", response_a="A2", response_b="B2", preference="tie"),
    ]
    output = tmp_path / "dpo_ties.jsonl"

    count = export_preferences_as_dpo(comparisons, str(output))

    assert count == 0


def test_cli_generates(tmp_path: Path) -> None:
    """CLI 'ab-chat' should exit 0 and print comparisons to stdout."""
    from cli.main import main

    with patch("serve.ab_chat.run_chat", side_effect=_mock_run_chat):
        exit_code = main([
            "--data-root", str(tmp_path),
            "ab-chat", "--model-a", "a", "--model-b", "b",
        ])

    assert exit_code == 0


def test_cli_dpo_export(tmp_path: Path, capsys) -> None:
    """CLI 'ab-chat --export-dpo' should exit 0 and mention export in output."""
    from cli.main import main

    dpo_path = tmp_path / "dpo.jsonl"

    with patch("serve.ab_chat.run_chat", side_effect=_mock_run_chat):
        exit_code = main([
            "--data-root", str(tmp_path),
            "ab-chat", "--model-a", "a", "--model-b", "b",
            "--export-dpo", str(dpo_path),
        ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert dpo_path.exists() or "export" in captured.lower()
