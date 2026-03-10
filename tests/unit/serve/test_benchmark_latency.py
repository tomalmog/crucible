"""Unit tests for latency profiling benchmark."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.benchmark_types import LatencyResult
from core.errors import CrucibleBenchmarkError
from serve.benchmark_latency import profile_latency


def _build_mock_torch() -> SimpleNamespace:
    """Build a minimal mock torch module for latency tests."""
    def tensor_fn(data, dtype=None):
        _ = dtype
        result = MagicMock()
        result.to = MagicMock(return_value=result)
        result.size = MagicMock(return_value=len(data[0]))
        return result

    torch_mock = SimpleNamespace(
        no_grad=MagicMock(return_value=MagicMock(
            __enter__=MagicMock(),
            __exit__=MagicMock(return_value=False),
        )),
        tensor=tensor_fn,
        long=0,
    )
    return torch_mock


def _build_mock_model() -> MagicMock:
    """Build a mock model for forward pass timing."""
    model = MagicMock()
    model.return_value = MagicMock()
    return model


def test_profile_latency_returns_result() -> None:
    """Latency profiling should return a LatencyResult."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3, 4, 5]]

    result = profile_latency(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
        num_runs=10,
    )

    assert isinstance(result, LatencyResult)
    assert result.mean_latency_ms >= 0
    assert result.p50_latency_ms >= 0
    assert result.p95_latency_ms >= 0
    assert result.p99_latency_ms >= 0
    assert result.throughput_tokens_per_sec >= 0


def test_profile_latency_raises_on_empty_sequences() -> None:
    """Latency profiling should raise on empty sequence list."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()

    with pytest.raises(CrucibleBenchmarkError, match="No sequences"):
        profile_latency(
            torch_module=torch_mock,
            model=model,
            sequences=[],
            device="cpu",
        )


def test_profile_latency_percentile_ordering() -> None:
    """Latency percentiles should satisfy p50 <= p95 <= p99."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3]]

    result = profile_latency(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
        num_runs=20,
    )

    assert result.p50_latency_ms <= result.p95_latency_ms
    assert result.p95_latency_ms <= result.p99_latency_ms


def test_profile_latency_sets_eval_mode() -> None:
    """Latency profiling should set model to eval mode."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3]]

    profile_latency(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
        num_runs=5,
    )

    model.eval.assert_called()


def test_profile_latency_calls_model_num_runs_times() -> None:
    """Latency profiling should call model exactly num_runs times."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3]]

    profile_latency(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
        num_runs=7,
    )

    assert model.call_count == 7
