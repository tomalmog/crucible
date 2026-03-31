"""Studio AI Agent — Claude API with Crucible MCP tools.

Agentic loop: call Claude, execute tools, repeat. Conversation persisted to disk.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

# ── Tool introspection ──────────────────────────────────────────────

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_TOOL_NAMES = [
    "list_datasets", "ingest_dataset", "delete_dataset", "push_dataset",
    "list_remote_datasets", "list_models", "register_model", "delete_model",
    "pull_model", "train", "submit_remote_training", "list_jobs",
    "job_status", "job_logs", "job_result", "cancel_job", "delete_job",
    "run_benchmark", "submit_remote_eval", "chat", "run_interp",
    "export_model", "merge_models", "list_clusters", "cluster_info",
    "hub_search_models", "hub_download_model", "run_sweep", "ab_chat",
    "lora_merge", "curate_dataset", "generate_synthetic_data",
    "hardware_profile",
]


def _parse_arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Extract per-parameter descriptions from an Args: docstring section."""
    if not docstring:
        return {}
    descs: dict[str, str] = {}
    in_args = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                break
            parts = stripped.split(":", 1)
            if len(parts) == 2 and parts[0].strip().isidentifier():
                descs[parts[0].strip()] = parts[1].strip()
    return descs


def _build_tool_definitions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Introspect MCP server functions to build Claude API tool schemas."""
    from serve import mcp_server

    tools: list[dict[str, Any]] = []
    registry: dict[str, Any] = {}

    for name in _TOOL_NAMES:
        fn = getattr(mcp_server, name, None)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        arg_descs = _parse_arg_descriptions(doc)
        description = doc.split("\n\n")[0].split("\nArgs:")[0].strip()

        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            ptype = _TYPE_MAP.get(param.annotation, "string")
            prop: dict[str, Any] = {"type": ptype}
            if pname in arg_descs:
                prop["description"] = arg_descs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        tools.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
        registry[name] = fn

    return tools, registry


_TOOLS: list[dict[str, Any]] | None = None
_REGISTRY: dict[str, Any] | None = None


def _get_tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global _TOOLS, _REGISTRY
    if _TOOLS is None:
        _TOOLS, _REGISTRY = _build_tool_definitions()
    return _TOOLS, _REGISTRY


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a Crucible tool by name and return the result string."""
    _, registry = _get_tools()
    fn = registry.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return str(fn(**tool_input))
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ── System prompt ───────────────────────────────────────────────────

_AGENT_BEHAVIOR = """
You are the Crucible Studio AI assistant. You help users train, evaluate,
and manage ML models through the Crucible platform.

CRITICAL: Do NOT use tools unless the user explicitly asks for data or
an action. If the user says "hello", "hey", or any greeting, just
respond conversationally. NEVER call a tool on a greeting.

Rules:
- Only use tools when the user asks a question that requires real data
  (e.g. "what models do I have?") or asks you to perform an action
  (e.g. "train a model", "run an eval").
- When using tools, check state first (list_models, list_datasets, list_jobs)
  before making changes.
- Explain what you're doing and why.
- When writing files, use the agent workspace directory by default.
- Return results in a clear, readable format.
- Be concise. Don't repeat tool output verbatim — summarize it.
- Never fabricate information. If you don't know something, say so.
- Never pretend to execute actions you didn't actually perform via tools.
"""


def _build_system_prompt(app_context: dict[str, Any], data_root: str) -> str:
    from serve.mcp_server import mcp
    workspace = str(Path(data_root) / "agent" / "workspace")
    os.makedirs(workspace, exist_ok=True)

    context_lines = [
        f"- Data root: {data_root}",
        f"- Agent workspace: {workspace}",
    ]
    if app_context.get("currentPage"):
        context_lines.append(f"- Current page: {app_context['currentPage']}")
    if app_context.get("selectedModel"):
        context_lines.append(f"- Selected model: {app_context['selectedModel']}")
    if app_context.get("selectedDataset"):
        context_lines.append(f"- Selected dataset: {app_context['selectedDataset']}")
    if app_context.get("modelNames"):
        context_lines.append(f"- Available models: {', '.join(app_context['modelNames'])}")
    if app_context.get("datasetNames"):
        context_lines.append(f"- Available datasets: {', '.join(app_context['datasetNames'])}")

    return (
        (mcp.instructions or "") + "\n\n"
        "## Current Studio State\n\n"
        + "\n".join(context_lines) + "\n\n"
        + _AGENT_BEHAVIOR
    )


# ── Conversation persistence ────────────────────────────────────────

def _load_conversation(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _save_conversation(path: Path, messages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": messages}, indent=2, default=str))


def _trim_conversation(messages: list[dict[str, Any]], max_chars: int = 400_000) -> list[dict[str, Any]]:
    """Trim oldest turns if conversation is too long (~100k tokens)."""
    total = len(json.dumps(messages))
    if total <= max_chars:
        return messages
    while len(messages) > 2 and len(json.dumps(messages)) > max_chars:
        messages.pop(0)
    return messages


# ── Agentic loop ────────────────────────────────────────────────────

_MAX_TOOL_LOOPS = 15
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_OLLAMA_MODEL = "llama3.1"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def run_agent_turn(
    conversation_path: Path,
    user_message: str,
    app_context: dict[str, Any],
    api_key: str,
    data_root: str,
    provider: str = "anthropic",
    model: str = "",
    ollama_url: str = "",
) -> dict[str, Any]:
    """Run a single agent conversation turn with tool use."""
    from serve.agent_backends import call_anthropic, call_ollama
    from serve.mcp_server import _ensure_backends
    _ensure_backends()

    tools, _ = _get_tools()
    system = _build_system_prompt(app_context, data_root)
    messages = _load_conversation(conversation_path)
    messages.append({"role": "user", "content": user_message})
    messages = _trim_conversation(messages)
    tools_used: list[str] = []

    for _ in range(_MAX_TOOL_LOOPS):
        if provider == "ollama":
            effective_model = model or _DEFAULT_OLLAMA_MODEL
            effective_url = ollama_url or _DEFAULT_OLLAMA_URL
            response = call_ollama(effective_url, effective_model, system, messages, tools)
        else:
            effective_model = model or _DEFAULT_ANTHROPIC_MODEL
            response = call_anthropic(api_key, effective_model, system, messages, tools)

        messages.append({"role": "assistant", "content": response.content_blocks})

        if response.stop_reason != "tool_use":
            break

        tool_results: list[dict[str, Any]] = []
        for block in response.content_blocks:
            if block.get("type") != "tool_use":
                continue
            tools_used.append(block["name"])
            result_str = execute_tool(block["name"], block.get("input", {}))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

    text_parts = [
        b.get("text", "") for b in response.content_blocks if b.get("type") == "text"
    ]
    _save_conversation(conversation_path, messages)

    return {
        "role": "assistant",
        "content": "\n".join(text_parts),
        "tools_used": tools_used,
    }


def load_conversation_for_display(conversation_path: Path) -> list[dict[str, Any]]:
    """Load conversation and convert to display format (role + content text)."""
    messages = _load_conversation(conversation_path)
    display: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                display.append({"role": "user", "content": content})
            # Skip tool_result user messages (they're not user-typed)
        elif msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, str):
                display.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_names = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_names.append(block.get("name", ""))
                if text_parts:
                    entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts),
                    }
                    if tool_names:
                        entry["tools_used"] = tool_names
                    display.append(entry)
    return display
