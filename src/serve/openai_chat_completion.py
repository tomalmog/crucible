"""OpenAI-compatible chat completions transport."""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.request import Request, urlopen


def _call_openai_compatible_chat(
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    headers: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Call an OpenAI-compatible chat completions endpoint."""
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": 4096}
    if tools:
        body["tools"] = tools

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers=request_headers,
    )
    with urlopen(request, timeout=300) as response:
        payload = json.loads(response.read())

    message = payload["choices"][0]["message"]
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if content:
        blocks.append({"type": "text", "text": content})

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        arguments = fn.get("arguments", "{}")
        blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id") or str(uuid.uuid4()),
            "name": fn.get("name", ""),
            "input": json.loads(arguments) if isinstance(arguments, str) else arguments,
        })

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return blocks, stop_reason
