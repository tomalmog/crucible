"""Unit tests for chat CLI command wiring."""

from __future__ import annotations

from cli.main import main
from core.chat_types import ChatResult
from store.dataset_sdk import CrucibleClient


def test_cli_chat_prints_response_text(monkeypatch, capsys) -> None:
    """Chat command should invoke chat and pass options correctly."""
    captured: dict[str, object] = {}

    def _fake_chat(self, options):
        captured["position_embedding_type"] = options.position_embedding_type
        captured["stream"] = options.stream
        return ChatResult(response_text="hello from model")

    monkeypatch.setattr(CrucibleClient, "chat", _fake_chat)
    args = [
        "chat",
        "--dataset",
        "demo",
        "--model-path",
        "./outputs/train/demo/model.pt",
        "--prompt",
        "hello",
        "--position-embedding-type",
        "sinusoidal",
    ]

    exit_code = main(args)
    _ = capsys.readouterr()

    assert (
        exit_code == 0
        and captured["position_embedding_type"] == "sinusoidal"
        and captured["stream"] is True
    )


def test_cli_chat_without_dataset_passes_none(monkeypatch, capsys) -> None:
    """Chat command should work without --dataset when tokenizer is available."""
    captured: dict[str, object] = {}

    def _fake_chat(self, options):
        captured["dataset_name"] = options.dataset_name
        captured["tokenizer_path"] = options.tokenizer_path
        return ChatResult(response_text="no dataset response")

    monkeypatch.setattr(CrucibleClient, "chat", _fake_chat)
    args = [
        "chat",
        "--model-path",
        "./outputs/train/demo/model.pt",
        "--prompt",
        "hello",
        "--tokenizer-path",
        "./outputs/train/demo/vocab.json",
    ]

    exit_code = main(args)
    _ = capsys.readouterr()

    assert (
        exit_code == 0
        and captured["dataset_name"] is None
        and captured["tokenizer_path"] == "./outputs/train/demo/vocab.json"
    )
