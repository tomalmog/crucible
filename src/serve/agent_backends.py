"""LLM backend adapters for the Studio AI Agent.

Each backend handles API calls and normalizes responses to a common format
so the agentic loop in studio_agent.py stays backend-agnostic.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, cast

from serve.openai_chat_adapter import OpenAiChatRequest, request_openai_chat_completion


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
        messages=cast(Any, messages),
        tools=cast(Any, tools),
    )
    blocks = [block.model_dump() for block in response.content]
    return LlmResponse(blocks, response.stop_reason or "end_turn")


def call_ollama(
    base_url: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> LlmResponse:
    """Call an Ollama-compatible OpenAI chat endpoint."""
    options = OpenAiChatRequest(
        endpoint_url=f"{base_url.rstrip('/')}/v1/chat/completions",
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        headers={},
        token_limit_parameter="max_tokens",
    )
    blocks, stop = request_openai_chat_completion(options)
    return LlmResponse(blocks, stop)


def call_openai(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> LlmResponse:
    """Call the OpenAI Chat Completions API."""
    options = OpenAiChatRequest(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    blocks, stop = request_openai_chat_completion(options)
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
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        return LlmResponse([], "end_turn")

    parts = candidates[0].content.parts or []
    for part in parts:
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
