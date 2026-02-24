"""Masked cross-entropy loss for SFT training.

This module provides a loss function that respects label masks,
computing loss only on response token positions while ignoring
prompt tokens marked with ignore_index=-100.
"""

from __future__ import annotations

from typing import Any

from serve.sft_tokenization import IGNORE_INDEX


def compute_sft_loss(
    torch_module: Any,
    logits: Any,
    labels: Any,
) -> Any:
    """Compute cross-entropy loss with prompt-masked label positions.

    Reshapes logits and labels for CrossEntropyLoss, using ignore_index
    to skip prompt token positions.

    Args:
        torch_module: Imported torch module.
        logits: Model output logits of shape (batch, seq_len, vocab).
        labels: Target labels of shape (batch, seq_len) with -100 masks.

    Returns:
        Scalar loss tensor.
    """
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    vocab_size = logits.shape[2]
    flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
    flat_labels = labels.reshape(batch_size * seq_len)
    return loss_fn(flat_logits, flat_labels)


def build_sft_loss_function(torch_module: Any) -> Any:
    """Build and return a CrossEntropyLoss configured for SFT masking.

    The returned loss function uses ignore_index=-100 to skip prompt
    token positions during backward pass computation.

    Args:
        torch_module: Imported torch module.

    Returns:
        Configured CrossEntropyLoss instance.
    """
    return torch_module.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
