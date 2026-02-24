"""Unit tests for drift detection during domain adaptation."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from serve.drift_detection import (
    DriftCheckResult,
    check_drift,
    compute_perplexity,
)


def _build_mock_torch() -> SimpleNamespace:
    """Build a minimal mock torch module for testing."""
    mock_loss_instance = MagicMock()
    mock_loss_instance.item.return_value = 1.0

    mock_loss_cls = MagicMock(return_value=mock_loss_instance)
    mock_loss_instance.return_value = mock_loss_instance
    mock_loss_instance.reshape.return_value = mock_loss_instance
    mock_loss_instance.__call__ = MagicMock(return_value=mock_loss_instance)

    mock_nn = SimpleNamespace(CrossEntropyLoss=mock_loss_cls)

    mock_tensor = MagicMock()
    mock_tensor.size.return_value = SimpleNamespace(__getitem__=lambda self, i: 100)

    def fake_tensor_fn(*args, **kwargs):
        return mock_tensor

    torch_module = SimpleNamespace(
        nn=mock_nn,
        no_grad=MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )),
        tensor=fake_tensor_fn,
        long=0,
    )
    return torch_module


class TestCheckDrift:
    """Tests for the check_drift threshold function."""

    def test_no_drift_within_threshold(self) -> None:
        """No drift when current is within ratio of baseline."""
        assert check_drift(10.0, 14.0, 1.5) is False

    def test_drift_detected_above_threshold(self) -> None:
        """Drift detected when current exceeds ratio of baseline."""
        assert check_drift(10.0, 16.0, 1.5) is True

    def test_drift_at_exact_threshold(self) -> None:
        """No drift when current equals ratio of baseline."""
        assert check_drift(10.0, 15.0, 1.5) is False

    def test_zero_baseline_no_drift(self) -> None:
        """Zero baseline should not trigger drift."""
        assert check_drift(0.0, 100.0, 1.5) is False

    def test_negative_baseline_no_drift(self) -> None:
        """Negative baseline should not trigger drift."""
        assert check_drift(-1.0, 100.0, 1.5) is False


class TestComputePerplexity:
    """Tests for perplexity computation."""

    def test_empty_sequences_returns_inf(self) -> None:
        """Empty sequence list should return infinite perplexity."""
        torch_module = _build_mock_torch()
        model = MagicMock()
        result = compute_perplexity(torch_module, model, [], "cpu")
        assert result == float("inf")

    def test_short_sequences_skipped(self) -> None:
        """Sequences with less than 2 tokens should be skipped."""
        torch_module = _build_mock_torch()
        model = MagicMock()
        result = compute_perplexity(torch_module, model, [[1]], "cpu")
        assert result == float("inf")


class TestDriftCheckResult:
    """Tests for DriftCheckResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        """DriftCheckResult should be immutable."""
        result = DriftCheckResult(
            epoch=1,
            perplexity=12.0,
            baseline_perplexity=10.0,
            drift_detected=True,
        )
        assert result.epoch == 1
        assert result.perplexity == 12.0
        assert result.baseline_perplexity == 10.0
        assert result.drift_detected is True
        with pytest.raises(AttributeError):
            result.epoch = 2  # type: ignore[misc]
