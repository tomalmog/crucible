"""Unit tests for perplexity benchmark computation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.benchmark_types import PerplexityResult
from core.errors import CrucibleBenchmarkError
from serve.benchmark_perplexity import compute_perplexity_benchmark


def _build_mock_torch() -> SimpleNamespace:
    """Build a minimal mock torch module for testing."""
    mock_tensor = MagicMock()
    mock_tensor.to = MagicMock(return_value=mock_tensor)
    mock_tensor.__getitem__ = MagicMock(return_value=mock_tensor)
    mock_tensor.size = MagicMock(return_value=10)
    mock_tensor.reshape = MagicMock(return_value=mock_tensor)

    non_pad_tensor = MagicMock()
    non_pad_tensor.sum.return_value = non_pad_tensor
    non_pad_tensor.item.return_value = 5
    mock_tensor.__ne__ = MagicMock(return_value=non_pad_tensor)

    loss_instance = MagicMock()
    loss_instance.return_value = MagicMock(item=MagicMock(return_value=5.0))

    nn_module = SimpleNamespace(
        CrossEntropyLoss=MagicMock(return_value=loss_instance),
    )

    def tensor_fn(data, dtype=None):
        _ = dtype
        result = MagicMock()
        result.to = MagicMock(return_value=result)
        result.__getitem__ = MagicMock(return_value=mock_tensor)
        return result

    torch_mock = SimpleNamespace(
        nn=nn_module,
        no_grad=MagicMock(return_value=MagicMock(
            __enter__=MagicMock(),
            __exit__=MagicMock(return_value=False),
        )),
        tensor=tensor_fn,
        long=0,
    )
    return torch_mock


def _build_mock_model() -> MagicMock:
    """Build a mock model that returns logits on forward pass."""
    model = MagicMock()
    logits = MagicMock()
    logits.size = MagicMock(return_value=10)
    logits.reshape = MagicMock(return_value=logits)
    model.return_value = logits
    return model


def test_compute_perplexity_returns_result() -> None:
    """Perplexity benchmark should return a PerplexityResult."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3, 4], [5, 6, 7, 8]]
    device = "cpu"

    result = compute_perplexity_benchmark(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device=device,
        batch_size=2,
    )

    assert isinstance(result, PerplexityResult)
    assert result.perplexity > 0
    assert result.num_sequences == 2


def test_compute_perplexity_raises_on_empty_sequences() -> None:
    """Perplexity benchmark should raise on empty sequence list."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()

    with pytest.raises(CrucibleBenchmarkError, match="No sequences"):
        compute_perplexity_benchmark(
            torch_module=torch_mock,
            model=model,
            sequences=[],
            device="cpu",
        )


def test_compute_perplexity_sets_eval_mode() -> None:
    """Perplexity benchmark should set model to eval mode."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2, 3]]

    compute_perplexity_benchmark(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
    )

    model.eval.assert_called()


def test_compute_perplexity_batches_sequences() -> None:
    """Perplexity benchmark should process sequences in batches."""
    torch_mock = _build_mock_torch()
    model = _build_mock_model()
    sequences = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    result = compute_perplexity_benchmark(
        torch_module=torch_mock,
        model=model,
        sequences=sequences,
        device="cpu",
        batch_size=2,
    )

    assert result.num_sequences == 5
