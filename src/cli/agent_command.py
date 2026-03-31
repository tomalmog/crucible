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
    conversation_path = Path(data_root) / "agent" / "conversation.json"

    action = payload.get("action", "chat")

    if action == "load":
        from serve.studio_agent import load_conversation_for_display
        messages = load_conversation_for_display(conversation_path)
        print(json.dumps({"messages": messages}))
        return 0

    if action == "clear":
        if conversation_path.exists():
            conversation_path.unlink()
        print(json.dumps({"messages": []}))
        return 0

    # action == "chat"
    provider = payload.get("provider", "anthropic")
    api_key = (
        payload.get("api_key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )
    if provider == "anthropic" and not api_key:
        print(json.dumps({
            "error": "Anthropic API key not configured. "
            "Set it in Settings or as ANTHROPIC_API_KEY environment variable.",
        }))
        return 1

    message = payload.get("message", "")
    if not message:
        print(json.dumps({"error": "No message provided."}))
        return 1

    from serve.studio_agent import run_agent_turn
    try:
        result = run_agent_turn(
            conversation_path=conversation_path,
            user_message=message,
            app_context=payload.get("context", {}),
            api_key=api_key,
            data_root=data_root,
            provider=payload.get("provider", "anthropic"),
            model=payload.get("model", ""),
            ollama_url=payload.get("ollama_url", ""),
        )
        print(json.dumps(result, default=str))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    return 0
