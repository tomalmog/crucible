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
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else "",
            }
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


# ── Gemini (Vertex AI) backend ─────────────────────────────────────


def _anthropic_tools_to_gemini(tools: list[dict[str, Any]]) -> list[Any]:
    """Convert Anthropic tool format to Gemini FunctionDeclaration format."""
    from google.genai import types

    declarations = []
    for t in tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters_json_schema=schema,
        ))
    return [types.Tool(function_declarations=declarations)]


def _anthropic_msgs_to_gemini(
    messages: list[dict[str, Any]],
) -> list[Any]:
    """Convert Anthropic message format to Gemini Content format."""
    from google.genai import types

    contents: list[Any] = []
    tool_id_to_name: dict[str, str] = {}

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and isinstance(content, str):
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=content)],
            ))
        elif role == "user" and isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    name = tool_id_to_name.get(tool_use_id, "unknown")
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        try:
                            response_data = json.loads(result_content)
                        except (json.JSONDecodeError, ValueError):
                            response_data = {"result": result_content}
                    else:
                        response_data = result_content
                    if not isinstance(response_data, dict):
                        response_data = {"result": response_data}
                    parts.append(types.Part(function_response=types.FunctionResponse(
                        name=name,
                        response=response_data,
                    )))
            if parts:
                contents.append(types.Content(role="user", parts=parts))
        elif role == "assistant" and isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    parts.append(types.Part(text=block["text"]))
                elif block.get("type") == "tool_use":
                    tool_id_to_name[block.get("id", "")] = block["name"]
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=block["name"],
                        args=block.get("input", {}),
                    )))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif role == "assistant" and isinstance(content, str):
            if content:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)],
                ))

    return contents


def call_gemini(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str = "",
) -> LlmResponse:
    """Call the Gemini API via the google-genai SDK.

    If api_key is provided, uses Google AI Studio (no GCP needed).
    Otherwise falls back to Vertex AI via application default credentials.
    """
    from google import genai
    from google.genai import types

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client(http_options=types.HttpOptions(api_version="v1"))

    gemini_contents = _anthropic_msgs_to_gemini(messages)
    gemini_tools = _anthropic_tools_to_gemini(tools) if tools else None

    response = client.models.generate_content(
        model=model,
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=4096,
            tools=gemini_tools,
        ),
    )

    blocks: list[dict[str, Any]] = []
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if part.text:
            blocks.append({"type": "text", "text": part.text})
        elif part.function_call:
            blocks.append({
                "type": "tool_use",
                "id": str(uuid.uuid4()),
                "name": part.function_call.name,
                "input": dict(part.function_call.args) if part.function_call.args else {},
            })

    stop = "tool_use" if any(b.get("type") == "tool_use" for b in blocks) else "end_turn"
    return LlmResponse(blocks, stop)
