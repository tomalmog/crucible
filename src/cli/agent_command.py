"""CLI wiring for the Studio AI agent chat command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from store.dataset_sdk import CrucibleClient


def add_agent_chat_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the agent-chat subcommand."""
    p = subparsers.add_parser("agent-chat", help="Run a Studio AI agent turn")
    p.add_argument(
        "--payload-file", required=True,
        help="Path to JSON file with action, message, context, api_key",
    )


def run_agent_chat_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle a single agent chat turn or management action."""
    payload = json.loads(Path(args.payload_file).read_text())
    data_root = str(client._config.data_root)
    agent_dir = Path(data_root) / "agent"

    action = payload.get("action", "chat")

    if action == "load":
        from serve.agent_chat_store import ensure_active_chat, list_chats
        from serve.studio_agent import load_conversation_for_display
        from serve.studio_agent import load_pending_chain
        session = ensure_active_chat(agent_dir, _optional_chat_id(payload))
        messages = load_conversation_for_display(session.conversation_path)
        print(json.dumps({
            "messages": messages,
            "active_chat_id": session.chat_id,
            "chats": list_chats(agent_dir),
            "chain": load_pending_chain(session.chat_dir),
        }))
        return 0

    if action == "list_chats":
        from serve.agent_chat_store import active_chat_id, list_chats
        print(json.dumps({
            "active_chat_id": active_chat_id(agent_dir),
            "chats": list_chats(agent_dir),
        }))
        return 0

    if action == "search_chats":
        from serve.agent_chat_store import active_chat_id, search_chats
        print(json.dumps({
            "active_chat_id": active_chat_id(agent_dir),
            "chats": search_chats(agent_dir, str(payload.get("query", ""))),
        }))
        return 0

    if action == "create_chat":
        from serve.agent_chat_store import create_chat, list_chats
        session = create_chat(agent_dir)
        print(json.dumps({
            "messages": [],
            "active_chat_id": session.chat_id,
            "chats": list_chats(agent_dir),
            "chain": None,
        }))
        return 0

    if action == "delete_chat":
        from serve.agent_chat_store import delete_chat, list_chats
        from serve.studio_agent import load_conversation_for_display, load_pending_chain
        session = delete_chat(agent_dir, str(payload.get("chat_id", "")))
        print(json.dumps({
            "messages": load_conversation_for_display(session.conversation_path),
            "active_chat_id": session.chat_id,
            "chats": list_chats(agent_dir),
            "chain": load_pending_chain(session.chat_dir),
        }))
        return 0

    if action == "clear":
        from serve.agent_chat_store import create_chat, list_chats
        session = create_chat(agent_dir)
        print(json.dumps({
            "messages": [],
            "active_chat_id": session.chat_id,
            "chats": list_chats(agent_dir),
            "chain": None,
        }))
        return 0

    if action == "load_chain":
        from serve.agent_chat_store import ensure_active_chat
        from serve.studio_agent import load_pending_chain
        session = ensure_active_chat(agent_dir, _optional_chat_id(payload))
        chain = load_pending_chain(session.chat_dir)
        print(json.dumps({"chain": chain}))
        return 0

    if action == "cancel_chain":
        from serve.agent_chat_store import ensure_active_chat
        from serve.studio_agent import delete_pending_chain
        session = ensure_active_chat(agent_dir, _optional_chat_id(payload))
        delete_pending_chain(session.chat_dir)
        print(json.dumps({"ok": True}))
        return 0

    # action == "chat"
    message = payload.get("message", "")
    if not message:
        print(json.dumps({"error": "No message provided."}))
        return 1

    from serve.studio_agent import is_medical_safety_demo_request
    is_demo_request = is_medical_safety_demo_request(message)

    provider = payload.get("provider", "anthropic")
    api_key = payload.get("api_key") or ""
    if provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    elif provider == "gemini":
        api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if provider == "anthropic" and not api_key and not is_demo_request:
        print(json.dumps({
            "error": "Anthropic API key not configured. "
            "Set it in Settings or as ANTHROPIC_API_KEY environment variable.",
        }))
        return 1
    if provider == "openai" and not api_key and not is_demo_request:
        print(json.dumps({
            "error": "OpenAI API key not configured. "
            "Set it in Settings or as OPENAI_API_KEY environment variable.",
        }))
        return 1

    from serve.agent_chat_store import ensure_active_chat, list_chats, refresh_chat_summary
    from serve.studio_agent import run_agent_turn
    from serve.agent_events import format_result_line, stdout_event_sink
    session = ensure_active_chat(agent_dir, _optional_chat_id(payload))
    try:
        result = run_agent_turn(
            conversation_path=session.conversation_path,
            user_message=message,
            app_context=payload.get("context", {}),
            api_key=api_key,
            data_root=data_root,
            provider=payload.get("provider", "anthropic"),
            model=payload.get("model", ""),
            ollama_url=payload.get("ollama_url", ""),
            event_sink=stdout_event_sink,
        )
        result["chat_id"] = session.chat_id
        result["chat"] = refresh_chat_summary(agent_dir, session.chat_id)
        result["chats"] = list_chats(agent_dir)
        print(format_result_line(result))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    return 0


def _optional_chat_id(payload: dict[str, object]) -> str | None:
    raw = payload.get("chat_id")
    return raw if isinstance(raw, str) and raw else None
