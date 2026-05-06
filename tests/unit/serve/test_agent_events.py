"""Unit tests for Studio agent event helpers."""

from __future__ import annotations

import json

from serve.agent_events import (
    STDOUT_RESULT_PREFIX,
    format_result_line,
    summarize_tool_input,
    summarize_tool_output,
)


def test_format_result_line_prefixes_json() -> None:
    """Final agent results should use the stdout result prefix."""
    line = format_result_line({"content": "Done", "tools_used": ["train"]})

    assert line.startswith(STDOUT_RESULT_PREFIX)
    assert json.loads(line[len(STDOUT_RESULT_PREFIX):]) == {
        "content": "Done",
        "tools_used": ["train"],
    }


def test_summarize_tool_output_prefers_key_fields() -> None:
    """Tool output summaries should highlight the useful keys."""
    summary = summarize_tool_output(
        json.dumps({
            "job_id": "job-123",
            "state": "running",
            "cluster": "gpu-box",
            "extra": "ignored",
        }),
    )

    assert "job-123" in summary
    assert "running" in summary
    assert "gpu-box" in summary


def test_summarize_tool_input_returns_single_line() -> None:
    """Tool input summaries should serialize arguments compactly."""
    summary = summarize_tool_input({"model_path": "/tmp/model.pt", "epochs": 3})

    assert "\n" not in summary
    assert "model.pt" in summary
    assert "epochs" in summary
