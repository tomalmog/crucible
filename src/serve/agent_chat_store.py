"""Persistent chat session store for the Studio agent."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentChatSession:
    """Resolved chat paths for one agent conversation."""

    chat_id: str
    chat_dir: Path
    conversation_path: Path


def ensure_active_chat(agent_dir: Path, chat_id: str | None = None) -> AgentChatSession:
    """Return an existing requested/active chat, creating one when needed."""
    _delete_legacy_global_files(agent_dir)
    index = _load_index(agent_dir)
    requested = _clean_chat_id(chat_id)
    if requested and _chat_dir(agent_dir, requested).exists():
        _set_active_chat_id(agent_dir, requested)
        return _session(agent_dir, requested)
    active = _clean_chat_id(str(index.get("active_chat_id", "")))
    if active and _chat_dir(agent_dir, active).exists():
        return _session(agent_dir, active)
    return create_chat(agent_dir)


def create_chat(agent_dir: Path, title: str = "New chat") -> AgentChatSession:
    """Create and activate a new empty chat."""
    chat_id = f"chat-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    chat_dir = _chat_dir(agent_dir, chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)
    _write_json(chat_dir / "conversation.json", {"messages": []})
    summary = {
        "id": chat_id,
        "title": title,
        "preview": "",
        "createdAt": now,
        "updatedAt": now,
        "messageCount": 0,
    }
    index = _load_index(agent_dir)
    chats = [summary, *_read_chat_summaries(index)]
    _save_index(agent_dir, {"active_chat_id": chat_id, "chats": chats})
    return _session(agent_dir, chat_id)


def delete_chat(agent_dir: Path, chat_id: str) -> AgentChatSession:
    """Delete a chat and return the next active chat session."""
    cleaned = _clean_chat_id(chat_id)
    if cleaned:
        shutil.rmtree(_chat_dir(agent_dir, cleaned), ignore_errors=True)
    index = _load_index(agent_dir)
    chats = [chat for chat in _read_chat_summaries(index) if chat.get("id") != cleaned]
    if not chats:
        _save_index(agent_dir, {"active_chat_id": "", "chats": []})
        return create_chat(agent_dir)
    active = str(chats[0]["id"])
    _save_index(agent_dir, {"active_chat_id": active, "chats": chats})
    return _session(agent_dir, active)


def list_chats(agent_dir: Path) -> list[dict[str, Any]]:
    """List chat summaries newest first."""
    index = _load_index(agent_dir)
    chats = _read_chat_summaries(index)
    chats.sort(key=lambda chat: str(chat.get("updatedAt", "")), reverse=True)
    return chats


def search_chats(agent_dir: Path, query: str) -> list[dict[str, Any]]:
    """Search chats by title, preview, and message text."""
    needle = " ".join(query.lower().split())
    if not needle:
        return list_chats(agent_dir)
    matches: list[dict[str, Any]] = []
    for chat in list_chats(agent_dir):
        chat_id = str(chat.get("id", ""))
        haystack = " ".join([
            str(chat.get("title", "")),
            str(chat.get("preview", "")),
            _conversation_search_text(_conversation_path(agent_dir, chat_id)),
        ]).lower()
        if needle in haystack:
            matches.append(chat)
    return matches


def refresh_chat_summary(agent_dir: Path, chat_id: str) -> dict[str, Any]:
    """Update title, preview, timestamp, and count from stored messages."""
    index = _load_index(agent_dir)
    chats = _read_chat_summaries(index)
    existing = next((chat for chat in chats if chat.get("id") == chat_id), {})
    created_at = str(existing.get("createdAt", _now_iso()))
    messages = _load_messages(_conversation_path(agent_dir, chat_id))
    summary = {
        "id": chat_id,
        "title": _infer_title(messages),
        "preview": _infer_preview(messages),
        "createdAt": created_at,
        "updatedAt": _now_iso(),
        "messageCount": len(_displayable_messages(messages)),
    }
    next_chats = [summary, *[chat for chat in chats if chat.get("id") != chat_id]]
    _save_index(agent_dir, {"active_chat_id": chat_id, "chats": next_chats})
    return summary


def active_chat_id(agent_dir: Path) -> str:
    """Return the active chat id, creating a chat if needed."""
    return ensure_active_chat(agent_dir).chat_id


def _session(agent_dir: Path, chat_id: str) -> AgentChatSession:
    chat_dir = _chat_dir(agent_dir, chat_id)
    return AgentChatSession(
        chat_id=chat_id,
        chat_dir=chat_dir,
        conversation_path=chat_dir / "conversation.json",
    )


def _set_active_chat_id(agent_dir: Path, chat_id: str) -> None:
    index = _load_index(agent_dir)
    _save_index(agent_dir, {"active_chat_id": chat_id, "chats": _read_chat_summaries(index)})


def _load_index(agent_dir: Path) -> dict[str, Any]:
    path = _index_path(agent_dir)
    if not path.exists():
        return {"active_chat_id": "", "chats": []}
    try:
        parsed = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"active_chat_id": "", "chats": []}
    return parsed if isinstance(parsed, dict) else {"active_chat_id": "", "chats": []}


def _save_index(agent_dir: Path, index: dict[str, Any]) -> None:
    _chats_root(agent_dir).mkdir(parents=True, exist_ok=True)
    _write_json(_index_path(agent_dir), index)


def _read_chat_summaries(index: dict[str, Any]) -> list[dict[str, Any]]:
    raw = index.get("chats", [])
    return [chat for chat in raw if isinstance(chat, dict)] if isinstance(raw, list) else []


def _load_messages(conversation_path: Path) -> list[dict[str, Any]]:
    if not conversation_path.exists():
        return []
    try:
        parsed = json.loads(conversation_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    raw = parsed.get("messages", []) if isinstance(parsed, dict) else []
    return [msg for msg in raw if isinstance(msg, dict)] if isinstance(raw, list) else []


def _displayable_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        msg for msg in messages
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), (str, list))
    ]


def _infer_title(messages: list[dict[str, Any]]) -> str:
    first_user = next(
        (str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"),
        "",
    ).strip()
    if not first_user:
        return "New chat"
    normalized = first_user.lower().replace("_", " ").replace("-", " ")
    if "medical assistant" in normalized and any(term in normalized for term in ("improve", "failing", "benchmark")):
        return "Improve medical-assistant"
    words = first_user.split()
    return " ".join(words[:7])[:64]


def _infer_preview(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(_displayable_messages(messages)):
        text = _message_text(msg).strip()
        if text:
            return text[:120]
    return ""


def _conversation_search_text(conversation_path: Path) -> str:
    return " ".join(_message_text(msg) for msg in _load_messages(conversation_path))


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return " ".join(parts)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def _delete_legacy_global_files(agent_dir: Path) -> None:
    for filename in ("conversation.json", "pending_chain.json"):
        path = agent_dir / filename
        if path.exists():
            path.unlink()


def _clean_chat_id(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value if ch.isalnum() or ch in "-_")


def _index_path(agent_dir: Path) -> Path:
    return _chats_root(agent_dir) / "index.json"


def _chats_root(agent_dir: Path) -> Path:
    return agent_dir / "chats"


def _chat_dir(agent_dir: Path, chat_id: str) -> Path:
    return _chats_root(agent_dir) / chat_id


def _conversation_path(agent_dir: Path, chat_id: str) -> Path:
    return _chat_dir(agent_dir, chat_id) / "conversation.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
