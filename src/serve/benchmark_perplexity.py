"""Perplexity benchmark computation for trained models.

This module evaluates model perplexity by running forward passes
over tokenized sequences and computing exp(avg cross-entropy).
"""

from __future__ import annotations

import math
from typing import Any

from core.benchmark_types import PerplexityResult
from core.errors import CrucibleBenchmarkError


def compute_perplexity_benchmark(
    torch_module: Any,
    model: Any,
    sequences: list[list[int]],
    device: Any,
    batch_size: int = 16,
) -> PerplexityResult:
    """Compute perplexity over tokenized sequences.

    Args:
        torch_module: Imported torch module.
        model: PyTorch model in eval mode.
        sequences: List of token-id sequences.
        device: Torch device for computation.
        batch_size: Number of sequences per forward pass.

    Returns:
        PerplexityResult with perplexity and token counts.

    Raises:
        CrucibleBenchmarkError: If no sequences or computation fails.
    """
    if not sequences:
        raise CrucibleBenchmarkError(
            "No sequences provided for perplexity benchmark."
        )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch_module.nn.CrossEntropyLoss(
        ignore_index=0, reduction="sum",
    )
    batches = _build_batches(sequences, batch_size)
    with torch_module.no_grad():
        for batch in batches:
            batch_loss, batch_tokens = _evaluate_batch(
                torch_module, model, batch, device, loss_fn,
            )
            total_loss += batch_loss
            total_tokens += batch_tokens
    if total_tokens == 0:
        raise CrucibleBenchmarkError(
            "No valid tokens found during perplexity evaluation."
        )
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100.0))
    return PerplexityResult(
        perplexity=perplexity,
        num_tokens=total_tokens,
        num_sequences=len(sequences),
    )


def _build_batches(
    sequences: list[list[int]], batch_size: int,
) -> list[list[list[int]]]:
    """Split sequences into fixed-size batches."""
    batches: list[list[list[int]]] = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i : i + batch_size])
    return batches


def _evaluate_batch(
    torch_module: Any,
    model: Any,
    batch: list[list[int]],
    device: Any,
    loss_fn: Any,
) -> tuple[float, int]:
    """Run one batch forward pass and return loss and token count."""
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [0] * (max_len - len(seq)) for seq in batch]
    input_tensor = torch_module.tensor(padded, dtype=torch_module.long)
    input_tensor = input_tensor.to(device)
    input_ids = input_tensor[:, :-1]
    target_ids = input_tensor[:, 1:]
    logits = model(input_ids)
    vocab_size = logits.size(-1)
    loss = loss_fn(
        logits.reshape(-1, vocab_size), target_ids.reshape(-1),
    )
    non_pad = (target_ids != 0).sum().item()
    return float(loss.item()), int(non_pad)
