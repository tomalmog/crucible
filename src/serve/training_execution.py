"""Training loop execution and checkpoint orchestration.

This module runs default and custom training loops for Forge serving.
It isolates epoch execution, resume loading, and checkpoint persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.errors import ForgeServeError
from core.types import BatchLossMetric, EpochMetric
from serve.custom_loop_loader import load_custom_training_loop
from serve.training_checkpoint import (
    ensure_checkpoint_dir,
    load_resume_checkpoint,
    prune_epoch_checkpoints,
    save_best_checkpoint,
    save_epoch_checkpoint,
)
from serve.training_context import TrainingRuntimeContext
from serve.training_epoch_pass import run_epoch_pass
from serve.training_hooks import invoke_hook
from serve.training_progress import TrainingProgressTracker, read_optimizer_learning_rate
from serve.training_run_types import TrainingRunState


@dataclass(frozen=True)
class TrainingLoopResult:
    """Outputs from one training loop execution."""

    epoch_metrics: list[EpochMetric]
    batch_metrics: list[BatchLossMetric]
    checkpoint_dir: Path | None
    best_checkpoint_path: Path | None
    resumed_from_checkpoint: str | None


@dataclass(frozen=True)
class ResumeTrainingState:
    """Resolved starting state when resuming from checkpoint."""

    next_epoch: int
    global_step: int
    best_validation_loss: float | None
    resumed_from_checkpoint: str | None


def run_training_loop(context: TrainingRuntimeContext) -> TrainingLoopResult:
    """Run default or custom training loop for a prepared runtime context."""
    resume_state = _resolve_resume_state(context)
    progress_tracker = _build_progress_tracker(context, resume_state)
    progress_tracker.log_training_started()
    custom_loop = load_custom_training_loop(context.options.custom_loop_path)
    if custom_loop is None:
        return _run_default_training_loop(context, resume_state, progress_tracker)
    metrics = custom_loop(context)
    _validate_metric_rows(metrics)
    return TrainingLoopResult(
        epoch_metrics=metrics,
        batch_metrics=[],
        checkpoint_dir=None,
        best_checkpoint_path=None,
        resumed_from_checkpoint=resume_state.resumed_from_checkpoint,
    )


def _build_progress_tracker(
    context: TrainingRuntimeContext,
    resume_state: ResumeTrainingState,
) -> TrainingProgressTracker:
    """Construct a progress tracker for this training execution."""
    return TrainingProgressTracker(
        dataset_name=context.options.dataset_name,
        total_epochs=context.options.epochs,
        start_epoch=resume_state.next_epoch,
        train_batch_count=len(context.train_batches),
        validation_batch_count=len(context.validation_batches),
        batch_log_interval_steps=context.options.progress_log_interval_steps,
    )


def _resolve_resume_state(context: TrainingRuntimeContext) -> ResumeTrainingState:
    """Resolve resume state and apply checkpoint payload when configured."""
    resume_path = context.options.resume_checkpoint_path
    if resume_path is None:
        return ResumeTrainingState(
            next_epoch=1,
            global_step=0,
            best_validation_loss=None,
            resumed_from_checkpoint=None,
        )
    resume_checkpoint = load_resume_checkpoint(
        checkpoint_path=resume_path,
        torch_module=context.torch_module,
        model=context.model,
        optimizer=context.optimizer,
        scheduler=context.scheduler,
        device=context.device,
    )
    if resume_checkpoint.next_epoch > context.options.epochs:
        raise ForgeServeError(
            f"Resume checkpoint epoch exceeds target epochs: start_epoch={resume_checkpoint.next_epoch}, "
            f"configured epochs={context.options.epochs}. Increase --epochs to continue training."
        )
    return ResumeTrainingState(
        next_epoch=resume_checkpoint.next_epoch,
        global_step=resume_checkpoint.global_step,
        best_validation_loss=resume_checkpoint.best_validation_loss,
        resumed_from_checkpoint=str(resume_checkpoint.checkpoint_path),
    )


def _run_default_training_loop(
    context: TrainingRuntimeContext,
    resume_state: ResumeTrainingState,
    progress_tracker: TrainingProgressTracker,
) -> TrainingLoopResult:
    """Run built-in epoch loop and persist periodic checkpoints."""
    epoch_rows: list[EpochMetric] = []
    batch_rows: list[BatchLossMetric] = []
    global_step = resume_state.global_step
    best_validation_loss = resume_state.best_validation_loss
    checkpoint_dir: Path | None = None
    best_checkpoint_path: Path | None = None
    try:
        for epoch_index in range(resume_state.next_epoch, context.options.epochs + 1):
            invoke_hook("on_epoch_start", context.hooks.on_epoch_start, context, epoch_index)
            train_loss, validation_loss, global_step = _run_epoch_cycle(
                context=context,
                progress_tracker=progress_tracker,
                epoch_index=epoch_index,
                global_step=global_step,
                batch_rows=batch_rows,
            )
            epoch_rows.append(
                EpochMetric(
                    epoch=epoch_index,
                    train_loss=round(train_loss, 6),
                    validation_loss=round(validation_loss, 6),
                )
            )
            _step_scheduler(context.scheduler)
            progress_tracker.log_epoch_completed(
                epoch_index=epoch_index,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=read_optimizer_learning_rate(context.optimizer),
            )
            invoke_hook(
                "on_epoch_end",
                context.hooks.on_epoch_end,
                context,
                epoch_index,
                train_loss,
                validation_loss,
            )
            _log_to_tracking_adapters(context, epoch_index, train_loss, validation_loss)
            (
                checkpoint_dir,
                best_checkpoint_path,
                best_validation_loss,
            ) = _persist_checkpoint_state(
                context=context,
                epoch_index=epoch_index,
                global_step=global_step,
                validation_loss=validation_loss,
                best_validation_loss=best_validation_loss,
                checkpoint_dir=checkpoint_dir,
                best_checkpoint_path=best_checkpoint_path,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        emergency_dir = checkpoint_dir or ensure_checkpoint_dir(context.output_dir)
        emergency_path = save_epoch_checkpoint(
            checkpoint_dir=emergency_dir,
            torch_module=context.torch_module,
            model=context.model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            epoch=epoch_index,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
        )
        checkpoint_dir = emergency_dir
        print(f"Emergency checkpoint saved: {emergency_path}")
        print(f"Resume with: --resume-checkpoint-path {emergency_path}")
        _transition_run_state(context, "failed")
    return TrainingLoopResult(
        epoch_metrics=epoch_rows,
        batch_metrics=batch_rows,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint_path=best_checkpoint_path,
        resumed_from_checkpoint=resume_state.resumed_from_checkpoint,
    )


def _run_epoch_cycle(
    context: TrainingRuntimeContext,
    progress_tracker: TrainingProgressTracker,
    epoch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
) -> tuple[float, float, int]:
    """Run train and validation passes for one epoch."""
    progress_tracker.log_epoch_started(epoch_index)
    train_loss, next_global_step = run_epoch_pass(
        context=context,
        batches=context.train_batches,
        phase="train",
        epoch_index=epoch_index,
        global_step=global_step,
        batch_rows=batch_rows,
        progress_tracker=progress_tracker,
    )
    validation_loss, next_global_step = run_epoch_pass(
        context=context,
        batches=context.validation_batches,
        phase="validation",
        epoch_index=epoch_index,
        global_step=next_global_step,
        batch_rows=batch_rows,
        progress_tracker=progress_tracker,
    )
    return train_loss, validation_loss, next_global_step


def _step_scheduler(scheduler: Any | None) -> None:
    """Advance optional scheduler by one epoch."""
    if scheduler is None:
        return
    scheduler.step()


def _persist_checkpoint_state(
    context: TrainingRuntimeContext,
    epoch_index: int,
    global_step: int,
    validation_loss: float,
    best_validation_loss: float | None,
    checkpoint_dir: Path | None,
    best_checkpoint_path: Path | None,
) -> tuple[Path | None, Path | None, float | None]:
    """Persist periodic and best checkpoints for one finished epoch."""
    should_save_epoch = epoch_index % context.options.checkpoint_every_epochs == 0
    improved_best = best_validation_loss is None or validation_loss < best_validation_loss
    should_save_best = context.options.save_best_checkpoint and improved_best
    next_best_validation = validation_loss if improved_best else best_validation_loss
    if not should_save_epoch and not should_save_best:
        return checkpoint_dir, best_checkpoint_path, next_best_validation
    _transition_run_state(context, "checkpointing")
    resolved_checkpoint_dir = checkpoint_dir or ensure_checkpoint_dir(context.output_dir)
    if should_save_epoch:
        epoch_checkpoint_path = save_epoch_checkpoint(
            checkpoint_dir=resolved_checkpoint_dir,
            torch_module=context.torch_module,
            model=context.model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            epoch=epoch_index,
            global_step=global_step,
            best_validation_loss=next_best_validation,
        )
        invoke_hook(
            "on_checkpoint",
            context.hooks.on_checkpoint,
            context,
            epoch_index,
            str(epoch_checkpoint_path),
        )
        prune_epoch_checkpoints(
            checkpoint_dir=resolved_checkpoint_dir,
            max_files=context.options.max_checkpoint_files,
        )
    updated_best_path = best_checkpoint_path
    if should_save_best:
        updated_best_path = save_best_checkpoint(
            checkpoint_dir=resolved_checkpoint_dir,
            torch_module=context.torch_module,
            model=context.model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            epoch=epoch_index,
            global_step=global_step,
            best_validation_loss=validation_loss,
        )
        invoke_hook(
            "on_checkpoint",
            context.hooks.on_checkpoint,
            context,
            epoch_index,
            str(updated_best_path),
        )
    _transition_run_state(context, "running")
    return resolved_checkpoint_dir, updated_best_path, next_best_validation


def _validate_metric_rows(metrics: list[EpochMetric]) -> None:
    """Ensure custom loops return at least one epoch metric row."""
    if not metrics:
        raise ForgeServeError(
            "Custom loop returned no metrics. Return at least one EpochMetric row."
        )


def _transition_run_state(context: TrainingRuntimeContext, state: TrainingRunState) -> None:
    """Transition lifecycle state when registry context exists."""
    if context.run_registry is None or context.run_id is None:
        return
    context.run_registry.transition(context.run_id, state)


def _log_to_tracking_adapters(
    context: TrainingRuntimeContext,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    """Send epoch metrics to any configured tracking adapters."""
    adapters = context.telemetry_collector
    if not adapters or not isinstance(adapters, list):
        return
    metrics = {"train_loss": train_loss, "validation_loss": validation_loss}
    for adapter in adapters:
        adapter.log_metrics(epoch, metrics)
