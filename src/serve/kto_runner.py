"""KTO training runner for Kahneman-Tversky Optimization.

This module orchestrates KTO training: loads unpaired preference data,
builds asymmetric loss, trains policy model, and persists artifacts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleKtoError, CrucibleServeError
from core.kto_types import KtoOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.hf_model_loader import build_or_load_model
from serve.device_selection import resolve_execution_device
from serve.kto_data_loader import load_kto_examples
from serve.kto_loss import build_kto_loss_function
from serve.model_weights import load_initial_weights
from serve.training_artifacts import ensure_training_output_dir
from serve.training_config_hash import compute_training_config_hash
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import run_training_loop
from serve.training_hooks import invoke_hook, load_training_hooks
from serve.training_optimization import build_training_optimization
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options


def run_kto_training(
    records: list[DataRecord],
    options: KtoOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full KTO training workflow and persist run lifecycle metadata."""
    training_options = _kto_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.initial_weights_path,
        config_hash=config_hash,
    )
    context: TrainingRuntimeContext | None = None
    try:
        context = _build_kto_runtime_context(
            records=records, options=options, training_options=training_options,
            random_seed=random_seed, run_id=run_record.run_id,
            config_hash=config_hash,
            run_registry=run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = _persist_kto_outputs(
            context=context, loop_result=loop_result, run_id=run_record.run_id,
            config_hash=config_hash,
            random_seed=random_seed,
        )
        invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
        run_registry.transition(
            run_id=run_record.run_id, next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        if context is not None:
            try:
                invoke_hook("on_run_error", context.hooks.on_run_error, context, str(error))
            except CrucibleServeError:
                pass
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _build_kto_runtime_context(
    records: list[DataRecord],
    options: KtoOptions,
    training_options: TrainingOptions,
    random_seed: int,
    run_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build runtime context for KTO training."""
    torch_module = _import_torch()
    validate_training_options(training_options)
    validate_file_paths(
        initial_weights_path=options.initial_weights_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options, base_model=options.base_model)
    kto_examples = load_kto_examples(options.kto_data_path)
    if not kto_examples:
        raise CrucibleKtoError(
            "No KTO examples loaded. Check the kto_data_path file content."
        )
    random.Random(random_seed).shuffle(kto_examples)
    train_batches, val_batches = _build_kto_batches(
        examples=kto_examples, tokenizer=tokenizer, options=options,
    )
    device = resolve_execution_device(torch_module)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=options.base_model,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, len(tokenizer.vocabulary)),
        device=device,
        initial_weights_path=options.initial_weights_path if not options.base_model else None,
        training_options=training_options,
    )
    precision_runtime = build_training_precision_runtime(
        torch_module=torch_module, requested_mode=options.precision_mode, device=device,
    )
    optimization = build_training_optimization(torch_module, model, training_options)
    hooks = load_training_hooks(options.hooks_path)
    loss_fn = build_kto_loss_function(
        torch_module, options.beta, options.desirable_weight, options.undesirable_weight,
    )
    return TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision_runtime, loss_function=loss_fn,
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device,
        run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )


def _build_kto_batches(
    examples: list[Any],
    tokenizer: Any,
    options: KtoOptions,
) -> tuple[list[Any], list[Any]]:
    """Build train/val batches from KTO examples."""
    from serve.tokenization import SequenceBatch

    split_idx = max(1, int(len(examples) * (1.0 - options.validation_split)))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:] if split_idx < len(examples) else []

    def to_batches(exs: list[Any]) -> list[Any]:
        batches = []
        for i in range(0, len(exs), options.batch_size):
            batch = exs[i : i + options.batch_size]
            all_inputs = []
            all_targets = []
            for ex in batch:
                text = ex.prompt + " " + ex.response
                ids = tokenizer.encode(text, options.max_token_length)
                padded = ids + [0] * (options.max_token_length - len(ids))
                all_inputs.append(padded)
                if ex.is_desirable:
                    all_targets.append(list(padded))
                else:
                    # Mark undesirable targets as -1 so the loss function
                    # can detect them and flip the gradient direction.
                    all_targets.append([-1] * len(padded))
            batches.append(SequenceBatch(inputs=all_inputs, targets=all_targets))
        return batches

    return to_batches(train_examples), to_batches(val_examples)


def _persist_kto_outputs(
    context: TrainingRuntimeContext,
    loop_result: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist KTO training outputs."""
    from dataclasses import asdict

    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import save_model_weights, save_training_history, save_training_plot
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_reproducibility_bundle import save_reproducibility_bundle

    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    config_path = save_training_config(context.output_dir, context.options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(
        context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
    )
    try:
        plot_path = save_training_plot(
            context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
        )
    except Exception:
        plot_path = None
    save_reproducibility_bundle(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.options.dataset_name,
        config_hash=config_hash,
        random_seed=random_seed, training_options=asdict(context.options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path), history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        checkpoint_dir=str(loop_result.checkpoint_dir) if loop_result.checkpoint_dir else None,
        best_checkpoint_path=str(loop_result.best_checkpoint_path) if loop_result.best_checkpoint_path else None,
        run_id=run_id, artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.options.dataset_name,
        parent_model_path=context.options.initial_weights_path,
        config_hash=config_hash, result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path="",
    )
    return TrainingRunResult(
        model_path=base_result.model_path, history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _kto_options_to_training_options(options: KtoOptions) -> TrainingOptions:
    """Map KtoOptions to TrainingOptions."""
    return TrainingOptions(
        dataset_name=options.dataset_name, output_dir=options.output_dir,
        epochs=options.epochs,
        learning_rate=options.learning_rate, batch_size=options.batch_size,
        max_token_length=options.max_token_length, validation_split=options.validation_split,
        precision_mode=options.precision_mode, optimizer_type=options.optimizer_type,
        weight_decay=options.weight_decay, hidden_dim=options.hidden_dim,
        num_layers=options.num_layers, attention_heads=options.attention_heads,
        mlp_hidden_dim=options.mlp_hidden_dim, mlp_layers=options.mlp_layers,
        hooks_path=options.hooks_path, initial_weights_path=options.initial_weights_path,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        save_best_checkpoint=options.save_best_checkpoint,
        progress_log_interval_steps=options.progress_log_interval_steps,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
    )


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "KTO training requires torch. Install torch to run crucible kto-train."
        ) from error
    return torch
