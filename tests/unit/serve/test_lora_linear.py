"""Unit tests for LoRA linear layer implementation."""

from __future__ import annotations

from serve.lora_linear import is_lora_linear


class _FakeModule:
    """Fake module with LoRA attributes."""

    def __init__(self, has_lora: bool) -> None:
        if has_lora:
            self.lora_a = "param_a"
            self.lora_b = "param_b"


def test_is_lora_linear_returns_true_for_lora_module() -> None:
    """Module with lora_a and lora_b should be detected as LoRA."""
    module = _FakeModule(has_lora=True)
    assert is_lora_linear(module) is True


def test_is_lora_linear_returns_false_for_plain_module() -> None:
    """Module without LoRA attributes should not be detected."""
    module = _FakeModule(has_lora=False)
    assert is_lora_linear(module) is False
