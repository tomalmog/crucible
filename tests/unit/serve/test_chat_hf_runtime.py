"""Tests for HuggingFace chat runtime policy."""

from __future__ import annotations

import os

from pytest import MonkeyPatch

from serve.chat_hf_runtime import quiet_hf_loading, resolve_hf_inference_device


def test_resolve_hf_inference_device_uses_cpu_for_mps(monkeypatch: MonkeyPatch) -> None:
    torch_module = _FakeTorch()

    monkeypatch.setattr("serve.chat_hf_runtime.resolve_execution_device", lambda _: _FakeDevice("mps"))

    device = resolve_hf_inference_device(torch_module)

    assert str(device) == "cpu"


def test_quiet_hf_loading_restores_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "old")
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)

    with quiet_hf_loading():
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "old"
    assert "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ


class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.type = name

    def __str__(self) -> str:
        return self.type


class _FakeTorch:
    def device(self, name: str) -> _FakeDevice:
        return _FakeDevice(name)
