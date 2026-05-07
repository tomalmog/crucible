"""Tests for OpenAI provider routing in the Studio agent loop."""

from __future__ import annotations

from typing import Any

import pytest

from serve import studio_agent
from serve.agent_backends import LlmResponse


def test_run_agent_turn_uses_openai_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_call_openai(
        api_key: str,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LlmResponse:
        captured["api_key"] = api_key
        captured["model"] = model
        return LlmResponse([{"type": "text", "text": "ok"}], "end_turn")

    monkeypatch.setattr("serve.agent_backends.call_openai", fake_call_openai)
    monkeypatch.setattr("serve.mcp_server._ensure_backends", lambda: None)
    monkeypatch.setattr(studio_agent, "_get_tools", lambda: ([], {}))
    monkeypatch.setattr(studio_agent, "_build_system_prompt", lambda app_context, data_root: "system")

    result = studio_agent.run_agent_turn(
        conversation_path=tmp_path / "conversation.json",
        user_message="hello",
        app_context={},
        api_key="sk-test",
        data_root=str(tmp_path),
        provider="openai",
    )

    assert (captured["api_key"], captured["model"], result["content"]) == (
        "sk-test",
        "gpt-5.4-mini",
        "ok",
    )
