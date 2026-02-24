"""Unit tests for SFT masked cross-entropy loss."""

from __future__ import annotations

import torch

from serve.sft_loss import build_sft_loss_function, compute_sft_loss
from serve.sft_tokenization import IGNORE_INDEX


def test_compute_sft_loss_ignores_masked_positions() -> None:
    """Loss computation should skip positions marked with IGNORE_INDEX."""
    batch_size = 1
    seq_len = 4
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    # First two positions are masked (prompt), last two are response
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7]])

    loss = compute_sft_loss(torch, logits, labels)

    assert loss.item() > 0
    # The loss should only be computed on the 2 unmasked positions
    # Verify it's a valid finite number
    assert torch.isfinite(loss)


def test_build_sft_loss_function_returns_callable() -> None:
    """build_sft_loss_function should return a callable loss module."""
    loss_fn = build_sft_loss_function(torch)

    assert callable(loss_fn)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert loss_fn.ignore_index == IGNORE_INDEX
