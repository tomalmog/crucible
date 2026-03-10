"""Unit tests for training hook loading and invocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleServeError
from serve.training_hooks import (
    build_loss_function_from_hooks,
    invoke_hook,
    load_training_hooks,
)


class _FakeTorch:
    class nn:
        @staticmethod
        def CrossEntropyLoss(ignore_index: int) -> str:
            return f"default-loss-{ignore_index}"


def test_load_training_hooks_reads_optional_callbacks(tmp_path: Path) -> None:
    """Hook loader should return callables from a valid Python module file."""
    hooks_file = tmp_path / "hooks.py"
    hooks_file.write_text(
        "def on_run_start(context):\n"
        "    _ = context\n"
        "def build_loss_function(context, torch_module):\n"
        "    _ = context\n"
        "    return 'custom-loss'\n",
        encoding="utf-8",
    )
    hooks = load_training_hooks(str(hooks_file))

    assert hooks.on_run_start is not None and hooks.build_loss_function is not None


def test_build_loss_function_from_hooks_uses_default_when_unset() -> None:
    """Missing loss hook should fall back to default cross-entropy loss."""
    hooks = load_training_hooks(None)
    loss = build_loss_function_from_hooks(_FakeTorch(), hooks, runtime_context=object())

    assert loss == "default-loss-0"


def test_invoke_hook_raises_traceable_error_on_callback_failure() -> None:
    """Hook invocation should wrap user callback failures in CrucibleServeError."""

    def _broken_hook() -> None:
        raise RuntimeError("boom")

    with pytest.raises(CrucibleServeError):
        invoke_hook("on_run_start", _broken_hook)

    assert True
