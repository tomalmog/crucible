"""Epoch-pass execution helpers for training loops.

This module runs train/validation passes over sequence batches and emits
structured batch-level progress updates.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from core.types import BatchLossMetric
from serve.gradient_accumulation import run_accumulated_batch_step
from serve.tokenization import SequenceBatch
from serve.training_context import TrainingRuntimeContext
from serve.training_hooks import invoke_hook
from serve.training_progress import TrainingProgressTracker


def run_epoch_pass(
    context: TrainingRuntimeContext,
    batches: list[SequenceBatch],
    phase: str,
    epoch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
    progress_tracker: TrainingProgressTracker,
) -> tuple[float, int]:
    """Run one full pass over train or validation batches."""
    if not batches:
        return 0.0, global_step
    training = phase == "train"
    accumulation_steps = context.gradient_accumulation_steps
    use_accumulation = training and accumulation_steps > 1
    if use_accumulation:
        return _run_accumulated_pass(
            context=context,
            batches=batches,
            phase=phase,
            epoch_index=epoch_index,
            global_step=global_step,
            batch_rows=batch_rows,
            progress_tracker=progress_tracker,
            accumulation_steps=accumulation_steps,
        )
    return _run_standard_pass(
        context=context,
        batches=batches,
        phase=phase,
        training=training,
        epoch_index=epoch_index,
        global_step=global_step,
        batch_rows=batch_rows,
        progress_tracker=progress_tracker,
    )


def _run_standard_pass(
    context: TrainingRuntimeContext,
    batches: list[SequenceBatch],
    phase: str,
    training: bool,
    epoch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
    progress_tracker: TrainingProgressTracker,
) -> tuple[float, int]:
    """Run a standard (non-accumulated) epoch pass."""
    total_loss = 0.0
    context.model.train(mode=training)
    for batch_index, batch in enumerate(batches, start=1):
        loss_value, global_step = _run_batch_step(
            context=context,
            batch=batch,
            training=training,
            epoch_index=epoch_index,
            batch_index=batch_index,
            global_step=global_step,
            batch_rows=batch_rows,
        )
        total_loss += loss_value
        progress_tracker.log_batch_progress(
            phase=phase,
            epoch_index=epoch_index,
            batch_index=batch_index,
            total_batches=len(batches),
            global_step=global_step,
            loss=loss_value,
        )
    return total_loss / len(batches), global_step


def _run_accumulated_pass(
    context: TrainingRuntimeContext,
    batches: list[SequenceBatch],
    phase: str,
    epoch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
    progress_tracker: TrainingProgressTracker,
    accumulation_steps: int,
) -> tuple[float, int]:
    """Run a training pass with gradient accumulation."""
    total_loss = 0.0
    context.model.train(mode=True)
    context.optimizer.zero_grad()
    current_accumulation = 0
    for batch_index, batch in enumerate(batches, start=1):
        current_accumulation += 1
        loss_value, did_step = run_accumulated_batch_step(
            context=context,
            batch=batch,
            accumulation_steps=accumulation_steps,
            current_accumulation=current_accumulation,
        )
        total_loss += loss_value
        if did_step:
            global_step += 1
            batch_rows.append(
                BatchLossMetric(
                    epoch=epoch_index,
                    batch_index=batch_index,
                    global_step=global_step,
                    train_loss=round(loss_value, 6),
                )
            )
            invoke_hook(
                "on_batch_end",
                context.hooks.on_batch_end,
                context,
                "train",
                epoch_index,
                batch_index,
                global_step,
                loss_value,
            )
            current_accumulation = 0
        progress_tracker.log_batch_progress(
            phase=phase,
            epoch_index=epoch_index,
            batch_index=batch_index,
            total_batches=len(batches),
            global_step=global_step,
            loss=loss_value,
        )
    if current_accumulation > 0:
        if context.precision_runtime.scaler is not None:
            context.precision_runtime.scaler.step(context.optimizer)
            context.precision_runtime.scaler.update()
        else:
            context.optimizer.step()
        context.optimizer.zero_grad()
        global_step += 1
        batch_rows.append(
            BatchLossMetric(
                epoch=epoch_index,
                batch_index=len(batches),
                global_step=global_step,
                train_loss=round(loss_value, 6),
            )
        )
    return total_loss / len(batches), global_step


def _run_batch_step(
    context: TrainingRuntimeContext,
    batch: SequenceBatch,
    training: bool,
    epoch_index: int,
    batch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
) -> tuple[float, int]:
    """Run one batch step and return loss value and updated global step."""
    inputs, targets = _tensorize_batch(context, batch)
    with _autocast_context(context):
        logits = context.model(inputs)
        loss = context.loss_function(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
    loss_value = float(loss.item())
    if not training:
        invoke_hook(
            "on_batch_end",
            context.hooks.on_batch_end,
            context,
            "validation",
            epoch_index,
            batch_index,
            global_step,
            loss_value,
        )
        return loss_value, global_step
    context.optimizer.zero_grad()
    if context.precision_runtime.scaler is not None:
        context.precision_runtime.scaler.scale(loss).backward()
        context.precision_runtime.scaler.step(context.optimizer)
        context.precision_runtime.scaler.update()
    else:
        loss.backward()
        context.optimizer.step()
    next_global_step = global_step + 1
    batch_rows.append(
        BatchLossMetric(
            epoch=epoch_index,
            batch_index=batch_index,
            global_step=next_global_step,
            train_loss=round(loss_value, 6),
        )
    )
    invoke_hook(
        "on_batch_end",
        context.hooks.on_batch_end,
        context,
        "train",
        epoch_index,
        batch_index,
        next_global_step,
        loss_value,
    )
    return loss_value, next_global_step


def _tensorize_batch(
    context: TrainingRuntimeContext,
    batch: SequenceBatch,
) -> tuple[Any, Any]:
    """Convert batch lists into padded tensors on target device."""
    torch_module = context.torch_module
    max_length = max(len(sequence) for sequence in batch.inputs)
    padded_inputs = [_pad_sequence(sequence, max_length) for sequence in batch.inputs]
    padded_targets = [_pad_sequence(sequence, max_length) for sequence in batch.targets]
    input_tensor = torch_module.tensor(padded_inputs, dtype=torch_module.long).to(context.device)
    target_tensor = torch_module.tensor(padded_targets, dtype=torch_module.long).to(context.device)
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
