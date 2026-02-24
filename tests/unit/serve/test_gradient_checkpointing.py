"""Unit tests for gradient checkpointing activation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from serve.gradient_checkpointing import (
    apply_gradient_checkpointing,
    _find_checkpointable_layers,
    _has_sub_modules,
    _resolve_checkpoint_module,
)


# ---------------------------------------------------------------------------
# Fake torch stubs
# ---------------------------------------------------------------------------

class _LeafModule:
    """Simulates a leaf nn.Module (e.g. Linear) with no children."""

    def children(self) -> list[Any]:
        return []

    def forward(self, x: Any) -> Any:
        return x


class _CompoundModule:
    """Simulates a compound nn.Module containing sub-modules."""

    def __init__(self) -> None:
        self._children = [_LeafModule(), _LeafModule()]

    def children(self) -> list[Any]:
        return list(self._children)

    def forward(self, x: Any) -> Any:
        return x


class _FakeModel:
    """Top-level model with both compound and leaf children."""

    def __init__(self) -> None:
        self.block1 = _CompoundModule()
        self.block2 = _CompoundModule()
        self.head = _LeafModule()

    def children(self) -> list[Any]:
        return [self.block1, self.block2, self.head]


class _FakeCheckpointModule:
    """Mimics torch.utils.checkpoint."""

    @staticmethod
    def checkpoint(fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args)


class _FakeTorch:
    """Mimics torch with utils.checkpoint available."""

    class utils:
        checkpoint = _FakeCheckpointModule()


class _FakeTorchNoUtils:
    """Mimics torch without utils attribute."""
    pass


class _FakeTorchNoCheckpoint:
    """Mimics torch with utils but no checkpoint sub-module."""

    class utils:
        pass


# ---------------------------------------------------------------------------
# Tests for _resolve_checkpoint_module
# ---------------------------------------------------------------------------

def test_resolve_checkpoint_module_returns_module() -> None:
    """Should return the checkpoint sub-module when torch.utils.checkpoint exists."""
    result = _resolve_checkpoint_module(_FakeTorch())
    assert result is _FakeTorch.utils.checkpoint


def test_resolve_checkpoint_module_none_when_no_utils() -> None:
    """Should return None when torch has no utils attribute."""
    result = _resolve_checkpoint_module(_FakeTorchNoUtils())
    assert result is None


def test_resolve_checkpoint_module_none_when_no_checkpoint() -> None:
    """Should return None when torch.utils has no checkpoint attribute."""
    result = _resolve_checkpoint_module(_FakeTorchNoCheckpoint())
    assert result is None


# ---------------------------------------------------------------------------
# Tests for _has_sub_modules
# ---------------------------------------------------------------------------

def test_has_sub_modules_true_for_compound() -> None:
    """Compound module with children should return True."""
    assert _has_sub_modules(_CompoundModule()) is True


def test_has_sub_modules_false_for_leaf() -> None:
    """Leaf module with no children should return False."""
    assert _has_sub_modules(_LeafModule()) is False


# ---------------------------------------------------------------------------
# Tests for _find_checkpointable_layers
# ---------------------------------------------------------------------------

def test_find_checkpointable_layers_returns_compound_only() -> None:
    """Should select only compound children, skipping leaf modules."""
    model = _FakeModel()
    layers = _find_checkpointable_layers(model)

    assert len(layers) == 2
    assert model.block1 in layers
    assert model.block2 in layers
    assert model.head not in layers


def test_find_checkpointable_layers_empty_for_leaf_only_model() -> None:
    """Model with only leaf children should yield no checkpointable layers."""
    model = _LeafModule()
    layers = _find_checkpointable_layers(model)

    assert layers == []


# ---------------------------------------------------------------------------
# Tests for apply_gradient_checkpointing
# ---------------------------------------------------------------------------

def test_apply_wraps_compound_layer_forward() -> None:
    """Compound layer forward should be replaced with checkpointed version."""
    model = _FakeModel()
    original_block1_forward = model.block1.forward

    apply_gradient_checkpointing(_FakeTorch(), model)

    # Forward should now be a different function
    assert model.block1.forward is not original_block1_forward
    assert model.block2.forward is not _CompoundModule.forward


def test_apply_does_not_wrap_leaf_modules() -> None:
    """Leaf modules should keep their original forward method."""
    model = _FakeModel()

    apply_gradient_checkpointing(_FakeTorch(), model)

    # Leaf module forward should remain the original unbound class method
    assert model.head.forward.__func__ is _LeafModule.forward


def test_apply_wrapped_forward_calls_checkpoint_fn() -> None:
    """Wrapped forward should invoke torch checkpoint with original forward."""
    checkpoint_mock = MagicMock()
    checkpoint_mock.checkpoint = MagicMock(return_value="result")

    class _MockTorch:
        class utils:
            checkpoint = checkpoint_mock

    model = _FakeModel()
    apply_gradient_checkpointing(_MockTorch(), model)

    # Call the wrapped forward
    model.block1.forward("input_tensor")

    checkpoint_mock.checkpoint.assert_called_once()
    call_args = checkpoint_mock.checkpoint.call_args
    assert call_args[1]["use_reentrant"] is False


def test_apply_no_op_when_checkpoint_unavailable() -> None:
    """Should gracefully return without modifying model when checkpoint module missing."""
    model = _FakeModel()

    apply_gradient_checkpointing(_FakeTorchNoUtils(), model)

    # Forward should still be the original class method (not replaced)
    assert model.block1.forward.__func__ is _CompoundModule.forward


def test_apply_no_op_on_model_with_no_children() -> None:
    """Model with no children should not cause errors."""
    model = _LeafModule()

    apply_gradient_checkpointing(_FakeTorch(), model)

    # Should complete without error; forward unchanged
    assert model.forward is not None
