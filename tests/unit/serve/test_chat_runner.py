"""Unit tests for chat runner dependency behavior."""

from __future__ import annotations

import builtins

import pytest

from core.chat_types import ChatOptions
from core.errors import CrucibleDependencyError
from core.types import DataRecord, RecordMetadata
from serve.chat_runner import run_chat


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [DataRecord(record_id="id-1", text="alpha beta gamma", metadata=metadata)]


def test_run_chat_raises_without_torch(monkeypatch) -> None:
    """Chat should fail clearly when torch dependency is missing."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    options = ChatOptions(
        model_path="./outputs/train/demo/model.pt",
        prompt="hello",
        dataset_name="demo",
    )

    with pytest.raises(CrucibleDependencyError):
        run_chat(_build_records(), options)

    assert True


def test_run_chat_routes_onnx_models_to_onnx_runner(monkeypatch) -> None:
    """ONNX model paths should use ONNX runtime path instead of torch loading."""
    captured: dict[str, object] = {}

    def _fake_onnx_runner(records, options):
        captured["dataset"] = options.dataset_name
        captured["model_path"] = options.model_path
        _ = records
        return "onnx response"

    monkeypatch.setattr("serve.chat_runner.run_onnx_chat", _fake_onnx_runner)
    options = ChatOptions(
        model_path="./outputs/train/demo/model.onnx",
        prompt="hello",
        dataset_name="demo",
    )

    result = run_chat(_build_records(), options)

    assert (
        result.response_text == "onnx response"
        and captured["dataset"] == "demo"
        and str(captured["model_path"]).endswith(".onnx")
    )
