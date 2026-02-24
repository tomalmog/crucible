"""Unit tests for memory-aware batch size planning."""

from __future__ import annotations

from core.types import TrainingOptions
from serve.memory_aware_batching import (
    MemoryPlan,
    compute_gradient_accumulation_steps,
    plan_memory_aware_batching,
)


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeModel:
    def __call__(self, inputs: object) -> object:
        return inputs


class _FakeTorch:
    long = "long"

    class cuda:
        @staticmethod
        def empty_cache() -> None:
            pass

    @staticmethod
    def no_grad() -> object:
        class _NoGrad:
            def __enter__(self) -> None:
                pass

            def __exit__(self, *args: object) -> None:
                pass

        return _NoGrad()

    @staticmethod
    def randint(low: int, high: int, size: tuple[int, ...], device: object) -> list[int]:
        return [0] * size[0]


def test_compute_gradient_accumulation_steps_exact_division() -> None:
    """Exact division should return effective / micro with no rounding."""
    result = compute_gradient_accumulation_steps(
        effective_batch_size=32,
        micro_batch_size=8,
    )

    assert result == 4


def test_compute_gradient_accumulation_steps_remainder() -> None:
    """Non-exact division should ceil to cover the full effective batch."""
    result = compute_gradient_accumulation_steps(
        effective_batch_size=32,
        micro_batch_size=10,
    )

    assert result == 4


def test_compute_gradient_accumulation_steps_minimum_one() -> None:
    """Accumulation steps should never be less than 1."""
    result = compute_gradient_accumulation_steps(
        effective_batch_size=4,
        micro_batch_size=16,
    )

    assert result == 1


def test_plan_memory_aware_batching_cpu_skips_probe(tmp_path) -> None:
    """CPU device should skip GPU memory probing and return desired batch size."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        batch_size=32,
    )

    plan = plan_memory_aware_batching(
        torch_module=_FakeTorch(),
        model=_FakeModel(),
        device=_FakeDevice("cpu"),
        options=options,
    )

    assert (
        plan.effective_batch_size == 32
        and plan.micro_batch_size == 32
        and plan.gradient_accumulation_steps == 1
        and plan.memory_probed is False
    )


def test_memory_plan_frozen() -> None:
    """MemoryPlan should be immutable (frozen dataclass)."""
    plan = MemoryPlan(
        effective_batch_size=16,
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        memory_probed=True,
    )

    try:
        plan.micro_batch_size = 8  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True

    assert raised
