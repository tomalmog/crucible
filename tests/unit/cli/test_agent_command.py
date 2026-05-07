"""Tests for Studio agent CLI provider configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

import pytest

from cli.agent_command import run_agent_chat_command
from core.config import CrucibleConfig
from store.dataset_sdk import CrucibleClient


def _build_client(tmp_path) -> CrucibleClient:
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    return CrucibleClient(config)


def test_agent_chat_uses_openai_env_key(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"provider": "openai", "message": "hello"}))
    captured: dict[str, Any] = {}

    def fake_run_agent_turn(**kwargs: Any) -> dict[str, Any]:
        captured["api_key"] = kwargs["api_key"]
        return {"content": "ok"}

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr("serve.studio_agent.run_agent_turn", fake_run_agent_turn)

    exit_code = run_agent_chat_command(
        _build_client(tmp_path),
        argparse.Namespace(payload_file=str(payload_path)),
    )

    assert (exit_code, captured["api_key"]) == (0, "sk-env")


def test_agent_chat_openai_without_key_returns_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"provider": "openai", "message": "hello"}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = run_agent_chat_command(
        _build_client(tmp_path),
        argparse.Namespace(payload_file=str(payload_path)),
    )
    output = capsys.readouterr().out

    assert exit_code == 1 and "OpenAI API key not configured" in output
