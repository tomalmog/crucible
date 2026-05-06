"""Unit tests for Studio agent backend adapters."""

from __future__ import annotations

from typing import Any

import pytest

from serve import agent_backends


def test_call_openai_maps_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI responses should normalize into agent text and tool blocks."""
    captured: dict[str, Any] = {}

    def fake_call(
        url: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        headers: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        captured["url"] = url
        captured["model"] = model
        captured["messages"] = messages
        captured["tools"] = tools
        captured["headers"] = headers
        return (
            [
                {"type": "text", "text": "Ready"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "list_models",
                    "input": {"limit": 5},
                },
            ],
            "tool_use",
        )

    monkeypatch.setattr(agent_backends, "_call_openai_compatible_chat", fake_call)

    response = agent_backends.call_openai(
        api_key="test-key",
        model="gpt-4.1-mini",
        system="system prompt",
        messages=[{"role": "user", "content": "Show me my models"}],
        tools=[
            {
                "name": "list_models",
                "description": "List models",
                "input_schema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            },
        ],
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["messages"][0] == {"role": "system", "content": "system prompt"}
    assert captured["messages"][1] == {"role": "user", "content": "Show me my models"}
    assert captured["tools"][0]["function"]["name"] == "list_models"
    assert captured["headers"] == {"Authorization": "Bearer test-key"}
    assert response.stop_reason == "tool_use"
    assert response.content_blocks[0] == {"type": "text", "text": "Ready"}
    assert response.content_blocks[1]["name"] == "list_models"
    assert response.content_blocks[1]["input"] == {"limit": 5}
