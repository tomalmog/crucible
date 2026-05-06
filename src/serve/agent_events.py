"""Helpers for streaming Studio agent events to the UI."""

from __future__ import annotations

import json
from typing import Any, Callable


STDOUT_EVENT_PREFIX = "CRUCIBLE_AGENT_EVENT:"
STDOUT_RESULT_PREFIX = "CRUCIBLE_AGENT_RESULT:"
EventSink = Callable[[dict[str, Any]], None]
_SUMMARY_LIMIT = 240
_SUMMARY_KEYS = (
    "error",
    "job_id",
    "state",
    "status",
    "cluster",
    "model_name",
    "model_path",
    "history_path",
    "dataset_name",
    "dataset_path",
    "sae_path",
    "steering_vector_path",
    "average_score",
)


def emit_event(
    event_sink: EventSink | None,
    event_type: str,
    **payload: Any,
) -> None:
    """Send one structured event to the UI, if a sink is configured."""
    if event_sink is None:
        return
    event_sink({"type": event_type, **payload})


def summarize_tool_input(tool_input: dict[str, Any]) -> str:
    """Return a compact single-line summary of tool arguments."""
    if not tool_input:
        return "No arguments"
    return _truncate(json.dumps(tool_input, default=str, sort_keys=True))


def summarize_tool_output(result_text: str) -> str:
    """Return a compact single-line summary of a tool result."""
    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        return _truncate(result_text.strip() or "No output")
    if isinstance(parsed, list):
        return f"Returned {len(parsed)} item(s)"
    if isinstance(parsed, dict):
        keys = [key for key in _SUMMARY_KEYS if key in parsed]
        if keys:
            return _truncate(json.dumps({key: parsed[key] for key in keys}, default=str))
        return _truncate(json.dumps(parsed, default=str))
    return _truncate(str(parsed))


def format_result_line(result: dict[str, Any]) -> str:
    """Serialize a final agent result to one stdout line."""
    return f"{STDOUT_RESULT_PREFIX}{json.dumps(result, default=str)}"


def stdout_event_sink(event: dict[str, Any]) -> None:
    """Write one structured event to stdout for the Studio sidebar."""
    print(f"{STDOUT_EVENT_PREFIX}{json.dumps(event, default=str)}", flush=True)


def _truncate(text: str, limit: int = _SUMMARY_LIMIT) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1]}…"
