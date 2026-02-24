"""Unit tests for LoRA adapter injection."""

from __future__ import annotations

import pytest

from core.errors import ForgeLoraError
from core.lora_types import LoraConfig
from serve.lora_injection import freeze_base_parameters, collect_lora_parameters


class _FakeParam:
    """Fake parameter with requires_grad control."""

    def __init__(self, name: str, requires_grad: bool = True) -> None:
        self.name = name
        self.requires_grad = requires_grad


class _FakeModelForFreeze:
    """Fake model for testing freeze behavior."""

    def __init__(self) -> None:
        self._params = [
            _FakeParam("layer.weight"),
            _FakeParam("layer.bias"),
            _FakeParam("lora_a"),
            _FakeParam("lora_b"),
        ]

    def named_parameters(self) -> list[tuple[str, _FakeParam]]:
        return [(p.name, p) for p in self._params]

    def parameters(self) -> list[_FakeParam]:
        return self._params


def test_freeze_base_parameters_freezes_non_lora() -> None:
    """Non-LoRA parameters should have requires_grad=False after freeze."""
    model = _FakeModelForFreeze()
    frozen = freeze_base_parameters(model)
    assert frozen == 2
    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_collect_lora_parameters_returns_only_trainable() -> None:
    """Only parameters with requires_grad=True should be collected."""
    model = _FakeModelForFreeze()
    freeze_base_parameters(model)
    lora_params = collect_lora_parameters(model)
    assert len(lora_params) == 2
    assert all(p.requires_grad for p in lora_params)


def test_lora_config_frozen() -> None:
    """LoraConfig should be immutable."""
    config = LoraConfig(rank=16)
    with pytest.raises(AttributeError):
        config.rank = 32  # type: ignore[misc]
