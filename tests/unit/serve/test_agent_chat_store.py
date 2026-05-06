"""Unit tests for Studio agent chat session storage."""

from __future__ import annotations

import json
from pathlib import Path

from serve.agent_chat_store import (
    create_chat,
    delete_chat,
    ensure_active_chat,
    refresh_chat_summary,
    search_chats,
)


def test_ensure_active_chat_deletes_legacy_global_files(tmp_path: Path) -> None:
    """Creating the first chat should discard old single-conversation files."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "conversation.json").write_text("{}", encoding="utf-8")
    (agent_dir / "pending_chain.json").write_text("{}", encoding="utf-8")

    session = ensure_active_chat(agent_dir)

    assert (
        session.conversation_path.exists(),
        (agent_dir / "conversation.json").exists(),
        (agent_dir / "pending_chain.json").exists(),
    ) == (True, False, False)


def test_refresh_chat_summary_reads_user_title_and_latest_preview(tmp_path: Path) -> None:
    """Summaries should describe the saved messages."""
    agent_dir = tmp_path / "agent"
    session = create_chat(agent_dir)
    session.conversation_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "my medical-assistant model is failing relevant benchmarks",
                    },
                    {"role": "assistant", "content": "Baseline score is 42 percent."},
                ],
            },
        ),
        encoding="utf-8",
    )

    summary = refresh_chat_summary(agent_dir, session.chat_id)

    assert (
        summary["title"],
        summary["preview"],
        summary["messageCount"],
    ) == ("Improve medical-assistant", "Baseline score is 42 percent.", 2)


def test_search_chats_matches_persisted_message_text(tmp_path: Path) -> None:
    """Search should inspect stored message bodies, not just titles."""
    agent_dir = tmp_path / "agent"
    medical_session = create_chat(agent_dir)
    create_chat(agent_dir)
    medical_session.conversation_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "improve the triage model"},
                    {"role": "assistant", "content": "Medical safety improved."},
                ],
            },
        ),
        encoding="utf-8",
    )
    refresh_chat_summary(agent_dir, medical_session.chat_id)

    matches = search_chats(agent_dir, "safety")

    assert [chat["id"] for chat in matches] == [medical_session.chat_id]


def test_delete_chat_activates_remaining_chat(tmp_path: Path) -> None:
    """Deleting the active chat should leave the remaining chat selected."""
    agent_dir = tmp_path / "agent"
    remaining_session = create_chat(agent_dir)
    deleted_session = create_chat(agent_dir)

    next_session = delete_chat(agent_dir, deleted_session.chat_id)

    assert next_session.chat_id == remaining_session.chat_id
