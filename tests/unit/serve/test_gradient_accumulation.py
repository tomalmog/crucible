"""Unit tests for gradient accumulation batch stepping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve.gradient_accumulation import run_accumulated_batch_step
from serve.tokenization import SequenceBatch
from serve.training_context import TrainingRuntimeContext
from serve.training_hooks import TrainingHooks
from serve.training_precision import TrainingPrecisionRuntime


class _FakeTensor:
    def __init__(self, value: float) -> None:
        self._value = value
        self.shape = (1, 1, 10)

    def item(self) -> float:
        return self._value

    def reshape(self, *args: int) -> "_FakeTensor":
        return self

    def __truediv__(self, other: object) -> "_FakeTensor":
        if isinstance(other, (int, float)):
            return _FakeTensor(self._value / other)
        return self

    def backward(self) -> None:
        pass


class _FakeModel:
    def __init__(self) -> None:
        self.forward_count = 0

    def __call__(self, inputs: object) -> _FakeTensor:
        self.forward_count += 1
        return _FakeTensor(0.5)

    def train(self, mode: bool = True) -> None:
        pass

    def parameters(self) -> list[Any]:
        return []


class _FakeOptimizer:
    def __init__(self) -> None:
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        self.zero_grad_count += 1


class _FakeLossFunction:
    def __call__(self, logits: _FakeTensor, targets: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(0.5)


class _FakeNnUtils:
    @staticmethod
    def clip_grad_norm_(parameters: Any, max_norm: float) -> None:
        pass


class _FakeNn:
    utils = _FakeNnUtils()


class _FakeTorchModule:
    long = "long"
    nn = _FakeNn()

    @staticmethod
    def tensor(data: list[list[int]], dtype: str) -> "_FakeTorchTensor":
        return _FakeTorchTensor()


class _FakeTorchTensor:
    shape = (1, 3)

    def to(self, device: object) -> "_FakeTorchTensor":
        return self

    def reshape(self, *args: int) -> "_FakeTorchTensor":
        return self


def _build_test_context(
    optimizer: _FakeOptimizer | None = None,
) -> TrainingRuntimeContext:
    """Build a minimal TrainingRuntimeContext for testing."""
    from core.types import TrainingOptions

    options = TrainingOptions(dataset_name="test", output_dir="/tmp/test")
    precision = TrainingPrecisionRuntime(
        requested_mode="fp32",
        resolved_mode="fp32",
        autocast_enabled=False,
        autocast_dtype=None,
        scaler=None,
    )
    return TrainingRuntimeContext(
        torch_module=_FakeTorchModule(),
        model=_FakeModel(),
        optimizer=optimizer or _FakeOptimizer(),
        scheduler=None,
        precision_runtime=precision,
        loss_function=_FakeLossFunction(),
        train_batches=[],
        validation_batches=[],
        tokenizer=None,  # type: ignore[arg-type]
        options=options,
        output_dir=Path("/tmp/test"),
        device="cpu",
        run_id=None,
        config_hash="test",
        hooks=TrainingHooks(),
        run_registry=None,
        gradient_accumulation_steps=4,
    )


def _make_batch() -> SequenceBatch:
    """Create a minimal test batch."""
    return SequenceBatch(
        inputs=[[1, 2, 3]],
        targets=[[2, 3, 4]],
    )


def test_run_accumulated_batch_step_accumulates_without_stepping() -> None:
    """Before reaching the accumulation boundary, optimizer should not step."""
    optimizer = _FakeOptimizer()
    context = _build_test_context(optimizer=optimizer)
    batch = _make_batch()

    loss_value, did_step = run_accumulated_batch_step(
        context=context,
        batch=batch,
        accumulation_steps=4,
        current_accumulation=2,
    )

    assert (
        loss_value == 0.5
        and did_step is False
        and optimizer.step_count == 0
        and optimizer.zero_grad_count == 0
    )


def test_run_accumulated_batch_step_steps_at_boundary() -> None:
    """At the accumulation boundary, optimizer should step and zero_grad."""
    optimizer = _FakeOptimizer()
    context = _build_test_context(optimizer=optimizer)
    batch = _make_batch()

    loss_value, did_step = run_accumulated_batch_step(
        context=context,
        batch=batch,
        accumulation_steps=4,
        current_accumulation=4,
    )

    assert (
        loss_value == 0.5
        and did_step is True
        and optimizer.step_count == 1
        and optimizer.zero_grad_count == 1
    )
