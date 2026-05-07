"""OpenAI-compatible chat completion adapter for Studio agent backends."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.request import Request, urlopen

TokenLimitParameter = Literal["max_completion_tokens", "max_tokens"]
DEFAULT_MAX_COMPLETION_TOKENS = 4096


@dataclass(frozen=True)
class OpenAiChatRequest:
    """Inputs for an OpenAI-compatible chat completion request."""

    endpoint_url: str
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    headers: Mapping[str, str]
    token_limit_parameter: TokenLimitParameter = "max_completion_tokens"


def request_openai_chat_completion(
    options: OpenAiChatRequest,
) -> tuple[list[dict[str, Any]], str]:
    """Call an OpenAI-compatible chat endpoint and return Anthropic-style blocks."""
    body = _build_openai_chat_body(options)
    headers = {"Content-Type": "application/json", **options.headers}
    req = Request(
        options.endpoint_url,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    with urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return _parse_openai_chat_result(result)


def _build_openai_chat_body(options: OpenAiChatRequest) -> dict[str, Any]:
    openai_tools = _anthropic_tools_to_openai(options.tools)
    body: dict[str, Any] = {
        "model": options.model,
        "messages": _anthropic_msgs_to_openai(options.system, options.messages),
        options.token_limit_parameter: DEFAULT_MAX_COMPLETION_TOKENS,
    }
    if openai_tools:
        body["tools"] = openai_tools
    return body


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _anthropic_msgs_to_openai(
    system: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and isinstance(content, str):
            out.append({"role": "user", "content": content})
        elif role == "user" and isinstance(content, list):
            out.extend(_tool_result_messages(content))
        elif role == "assistant" and isinstance(content, list):
            out.append(_assistant_message(content))
        elif role == "assistant" and isinstance(content, str):
            out.append({"role": "assistant", "content": content})
    return out


def _tool_result_messages(content: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
            })
    return messages


def _assistant_message(content: list[Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
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
    entry: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


def _parse_openai_chat_result(result: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    choice = result["choices"][0]
    message = choice["message"]
    blocks: list[dict[str, Any]] = []

    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id") or str(uuid.uuid4()),
            "name": function.get("name", ""),
            "input": _parse_tool_arguments(function.get("arguments", "{}")),
        })

    stop = "tool_use" if tool_calls else "end_turn"
    return blocks, stop


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"raw_arguments": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}
