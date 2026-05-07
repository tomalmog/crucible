"""Tests for the OpenAI-compatible Studio agent chat adapter."""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any
from urllib.request import Request

import pytest

from serve.agent_backends import call_openai
from serve.openai_chat_adapter import OpenAiChatRequest, request_openai_chat_completion


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_call_openai_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeResponse:
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse({"choices": [{"message": {"content": "ready"}}]})

    monkeypatch.setattr("serve.openai_chat_adapter.urlopen", fake_urlopen)

    response = call_openai("sk-test", "gpt-4o-mini", "system", [], [])

    assert (captured["authorization"], response.content_blocks) == (
        "Bearer sk-test",
        [{"type": "text", "text": "ready"}],
    )


def test_request_openai_chat_completion_converts_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode() if request.data else "{}")
        return _FakeResponse({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "list_models",
                            "arguments": "{\"limit\": 5}",
                        },
                    }],
                },
            }],
        })

    options = OpenAiChatRequest(
        endpoint_url="http://localhost/v1/chat/completions",
        model="gpt-4o-mini",
        system="system",
        messages=[{"role": "user", "content": "list models"}],
        tools=[{
            "name": "list_models",
            "description": "List registered models.",
            "input_schema": {"type": "object", "properties": {}},
        }],
        headers={},
    )
    monkeypatch.setattr("serve.openai_chat_adapter.urlopen", fake_urlopen)

    blocks, stop = request_openai_chat_completion(options)

    assert (
        captured["body"]["tools"][0]["function"]["name"],
        captured["body"]["max_completion_tokens"],
        "max_tokens" in captured["body"],
        blocks,
        stop,
    ) == (
        "list_models",
        4096,
        False,
        [{"type": "tool_use", "id": "call-1", "name": "list_models", "input": {"limit": 5}}],
        "tool_use",
    )


def test_request_openai_chat_completion_supports_legacy_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode() if request.data else "{}")
        return _FakeResponse({"choices": [{"message": {"content": "ready"}}]})

    options = OpenAiChatRequest(
        endpoint_url="http://localhost/v1/chat/completions",
        model="llama3.1",
        system="system",
        messages=[],
        tools=[],
        headers={},
        token_limit_parameter="max_tokens",
    )
    monkeypatch.setattr("serve.openai_chat_adapter.urlopen", fake_urlopen)

    request_openai_chat_completion(options)

    assert (
        captured["body"]["max_tokens"],
        "max_completion_tokens" in captured["body"],
    ) == (4096, False)
