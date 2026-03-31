"""LLM backend adapters for the Studio AI Agent.

Each backend handles API calls and normalizes responses to a common format
so the agentic loop in studio_agent.py stays backend-agnostic.
"""
from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.request import urlopen, Request


class LlmResponse:
    """Normalized response from any LLM backend."""

    __slots__ = ("content_blocks", "stop_reason")

    def __init__(
        self, content_blocks: list[dict[str, Any]], stop_reason: str,
    ) -> None:
        self.content_blocks = content_blocks
        self.stop_reason = stop_reason


def call_anthropic(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> LlmResponse:
    """Call the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=messages,
        tools=tools,
    )
    blocks = [block.model_dump() for block in response.content]
    return LlmResponse(blocks, response.stop_reason or "end_turn")


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool format to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _anthropic_msgs_to_openai(
    system: str, messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI chat format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and isinstance(content, str):
            out.append({"role": "user", "content": content})
        elif role == "user" and isinstance(content, list):
            # Tool results
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
        elif role == "assistant" and isinstance(content, list):
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            entry: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                entry["content"] = "\n".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        elif role == "assistant" and isinstance(content, str):
            out.append({"role": "assistant", "content": content})
    return out


def call_ollama(
    base_url: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> LlmResponse:
    """Call an Ollama-compatible OpenAI chat endpoint."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    openai_messages = _anthropic_msgs_to_openai(system, messages)
    openai_tools = _anthropic_tools_to_openai(tools)

    body: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": 4096,
    }
    if openai_tools:
        body["tools"] = openai_tools

    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    choice = result["choices"][0]
    message = choice["message"]
    blocks: list[dict[str, Any]] = []

    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})

    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or str(uuid.uuid4()),
            "name": fn.get("name", ""),
            "input": json.loads(args) if isinstance(args, str) else args,
        })

    stop = "tool_use" if tool_calls else "end_turn"
    return LlmResponse(blocks, stop)
