"""Gradient accumulation for memory-efficient training.

This module provides a batch step that accumulates gradients over multiple
micro-batches before calling optimizer.step(), enabling larger effective
batch sizes on memory-constrained hardware.

Assumptions:
- The caller manages the accumulation counter across batch iterations.
- Loss is scaled by 1/accumulation_steps so the effective gradient matches
  a single large-batch step.
- Mixed precision (GradScaler) is handled correctly when present.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from core.types import BatchLossMetric
from serve.tokenization import SequenceBatch
from serve.training_context import TrainingRuntimeContext
from serve.training_hooks import invoke_hook


def run_accumulated_batch_step(
    context: TrainingRuntimeContext,
    batch: SequenceBatch,
    accumulation_steps: int,
    current_accumulation: int,
) -> tuple[float, bool]:
    """Run one micro-batch with gradient accumulation.

    Performs forward+backward with loss scaled by 1/accumulation_steps.
    Only calls optimizer.step() and zero_grad() when the accumulation
    boundary is reached (current_accumulation == accumulation_steps).

    Returns:
        Tuple of (loss_value, did_optimizer_step).

    Side-effects:
        Updates model gradients. Steps optimizer at accumulation boundary.
    """
    inputs, targets = _tensorize_batch(context, batch)
    with _autocast_context(context):
        logits = context.model(inputs)
        loss = context.loss_function(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
    scaled_loss = loss / accumulation_steps
    loss_value = float(loss.item())

    if context.precision_runtime.scaler is not None:
        context.precision_runtime.scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    did_step = current_accumulation == accumulation_steps
    if did_step:
        if context.precision_runtime.scaler is not None:
            context.precision_runtime.scaler.step(context.optimizer)
            context.precision_runtime.scaler.update()
        else:
            context.optimizer.step()
        context.optimizer.zero_grad()

    return loss_value, did_step


def _tensorize_batch(
    context: TrainingRuntimeContext,
    batch: SequenceBatch,
) -> tuple[Any, Any]:
    """Convert batch lists into padded tensors on target device."""
    torch_module = context.torch_module
    max_length = max(len(sequence) for sequence in batch.inputs)
    padded_inputs = [_pad_sequence(s, max_length) for s in batch.inputs]
    padded_targets = [_pad_sequence(s, max_length) for s in batch.targets]
    input_tensor = torch_module.tensor(
        padded_inputs, dtype=torch_module.long,
    ).to(context.device)
    target_tensor = torch_module.tensor(
        padded_targets, dtype=torch_module.long,
    ).to(context.device)
    return input_tensor, target_tensor


def _autocast_context(context: TrainingRuntimeContext) -> Any:
    """Build autocast context for mixed precision forward pass."""
    precision_runtime = context.precision_runtime
    if not precision_runtime.autocast_enabled:
        return nullcontext()
    autocast_fn = getattr(context.torch_module, "autocast", None)
    if autocast_fn is None:
        return nullcontext()
    return autocast_fn(
        device_type=getattr(context.device, "type", str(context.device)),
        dtype=precision_runtime.autocast_dtype,
    )


def _pad_sequence(sequence: list[int], max_length: int) -> list[int]:
    """Pad sequence with pad id 0 up to max length."""
    if len(sequence) >= max_length:
        return sequence
    return sequence + ([0] * (max_length - len(sequence)))
