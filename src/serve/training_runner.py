"""PyTorch training workflow orchestration."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeServeError
from core.types import BatchLossMetric, DataRecord, EpochMetric, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.gradient_checkpointing import apply_gradient_checkpointing
from serve.memory_aware_batching import plan_memory_aware_batching
from serve.model_weights import load_initial_weights
from serve.tokenization import (
    SequenceBatch,
    build_sequence_batches,
    build_training_sequences,
    split_sequences,
)
from serve.training_artifact_contract import save_training_artifact_contract
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_model_weights,
    save_training_history,
    save_training_plot,
)
from serve.training_config_hash import compute_training_config_hash
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import TrainingLoopResult, run_training_loop
from serve.training_hooks import (
    build_loss_function_from_hooks,
    invoke_hook,
    load_training_hooks,
)
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
from serve.training_optimization import build_training_optimization
from serve.training_precision import build_training_precision_runtime
from serve.training_reproducibility_bundle import save_reproducibility_bundle
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_training_options


def run_training(
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run a full training workflow and persist run lifecycle metadata."""
    config_hash = compute_training_config_hash(options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        dataset_version_id=dataset_version_id,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.initial_weights_path,
        config_hash=config_hash,
    )
    context: TrainingRuntimeContext | None = None
    try:
        context = _build_runtime_context(
            records=records,
            options=options,
            random_seed=random_seed,
            run_id=run_record.run_id,
            dataset_version_id=dataset_version_id,
            config_hash=config_hash,
            run_registry=run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = _persist_training_outputs(
            context=context,
            loop_result=loop_result,
            run_id=run_record.run_id,
            dataset_version_id=dataset_version_id,
            config_hash=config_hash,
            random_seed=random_seed,
        )
        invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
        run_registry.transition(
            run_id=run_record.run_id,
            next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        if context is not None:
            try:
                invoke_hook("on_run_error", context.hooks.on_run_error, context, str(error))
            except ForgeServeError:
                pass
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _build_runtime_context(
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
    run_id: str,
    dataset_version_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build an initialized runtime context from records and options."""
    torch_module = _import_torch()
    validate_training_options(options)
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, options)
    sequences = build_training_sequences(records, tokenizer, options.max_token_length)
    if not sequences:
        raise ForgeServeError(
            "No trainable sequences were generated from dataset records. "
            "Check dataset content and max token length."
        )
    random.Random(random_seed).shuffle(sequences)
    train_batches, validation_batches = _build_batches(sequences, options)
    model = load_training_model(torch_module, options, len(tokenizer.vocabulary))
    if options.gradient_checkpointing:
        apply_gradient_checkpointing(torch_module, model)
    device = resolve_execution_device(torch_module)
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.initial_weights_path,
        device=device,
    )
    precision_runtime = build_training_precision_runtime(
        torch_module=torch_module,
        requested_mode=options.precision_mode,
        device=device,
    )
    accumulation_steps = _resolve_accumulation_steps(
        torch_module=torch_module,
        model=model,
        device=device,
        options=options,
    )
    optimization = build_training_optimization(torch_module, model, options)
    hooks = load_training_hooks(options.hooks_path)
    context = TrainingRuntimeContext(
        torch_module=torch_module,
        model=model,
        optimizer=optimization.optimizer,
        scheduler=optimization.scheduler,
        precision_runtime=precision_runtime,
        loss_function=torch_module.nn.CrossEntropyLoss(ignore_index=0),
        train_batches=train_batches,
        validation_batches=validation_batches,
        tokenizer=tokenizer,
        options=options,
        output_dir=output_dir,
        device=device,
        run_id=run_id,
        dataset_version_id=dataset_version_id,
        config_hash=config_hash,
        hooks=hooks,
        run_registry=run_registry,
        gradient_accumulation_steps=accumulation_steps,
    )
    context.loss_function = build_loss_function_from_hooks(
        torch_module=torch_module,
        hooks=hooks,
        runtime_context=context,
    )
    return context


def _persist_training_outputs(
    context: TrainingRuntimeContext,
    loop_result: TrainingLoopResult,
    run_id: str,
    dataset_version_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist model/history/plot outputs and return summary metadata."""
    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    config_path = save_training_config(context.output_dir, context.options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(
        context.output_dir,
        loop_result.epoch_metrics,
        loop_result.batch_metrics,
    )
    plot_path = _try_save_plot(
        context.output_dir,
        loop_result.epoch_metrics,
        loop_result.batch_metrics,
    )
    reproducibility_bundle_path = save_reproducibility_bundle(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.options.dataset_name,
        dataset_version_id=dataset_version_id,
        config_hash=config_hash,
        random_seed=random_seed,
        training_options=asdict(context.options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        checkpoint_dir=str(loop_result.checkpoint_dir) if loop_result.checkpoint_dir else None,
        best_checkpoint_path=(
            str(loop_result.best_checkpoint_path) if loop_result.best_checkpoint_path else None
        ),
        resumed_from_checkpoint=loop_result.resumed_from_checkpoint,
        run_id=run_id,
        artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.options.dataset_name,
        dataset_version_id=dataset_version_id,
        parent_model_path=context.options.initial_weights_path,
        config_hash=config_hash,
        result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(reproducibility_bundle_path),
    )
    return TrainingRunResult(
        model_path=base_result.model_path,
        history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        resumed_from_checkpoint=base_result.resumed_from_checkpoint,
        run_id=run_id,
        artifact_contract_path=str(contract_path),
    )


def _import_torch() -> Any:
    """Import torch dependency used by training workflows."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Training requires torch, but it is not installed. Install torch to run forge train."
        ) from error
    return torch


def _build_batches(
    sequences: list[list[int]],
    options: TrainingOptions,
) -> tuple[list[SequenceBatch], list[SequenceBatch]]:
    """Build train and validation sequence batches from tokenized sequences."""
    train_sequences, validation_sequences = split_sequences(
        sequences,
        options.validation_split,
    )
    train_batches = build_sequence_batches(train_sequences, options.batch_size)
    validation_batches = build_sequence_batches(validation_sequences, options.batch_size)
    return train_batches, validation_batches


def _try_save_plot(
    output_dir: Path,
    epoch_metrics: list[EpochMetric],
    batch_metrics: list[BatchLossMetric],
) -> Path | None:
    """Save training plot unless plotting dependency is unavailable."""
    try:
        return save_training_plot(output_dir, epoch_metrics, batch_metrics)
    except ForgeDependencyError:
        return None


def _resolve_accumulation_steps(
    torch_module: Any,
    model: Any,
    device: Any,
    options: TrainingOptions,
) -> int:
    """Resolve gradient accumulation steps from options or memory planner."""
    if options.auto_micro_batch:
        memory_plan = plan_memory_aware_batching(
            torch_module=torch_module,
            model=model,
            device=device,
            options=options,
        )
        return memory_plan.gradient_accumulation_steps
    return options.gradient_accumulation_steps
