"""SFT training runner for supervised fine-tuning workflows.

This module orchestrates SFT training: loads data, tokenizes with masking,
builds model, runs training loop, and persists artifacts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleServeError, CrucibleSftError
from core.sft_types import SftOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import build_or_load_model
from serve.model_weights import load_initial_weights
from serve.sft_batch_builder import build_sft_batches, persist_sft_outputs
from serve.sft_data_loader import load_sft_examples
from serve.sft_loss import build_sft_loss_function
from serve.sft_tokenization import build_sft_sequences, pack_sft_sequences
from serve.training_artifacts import ensure_training_output_dir
from serve.training_config_hash import compute_training_config_hash
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import run_training_loop
from serve.training_hooks import invoke_hook, load_training_hooks
from serve.training_optimization import build_training_optimization
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options


def run_sft_training(
    records: list[DataRecord],
    options: SftOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full SFT training workflow and persist run lifecycle metadata."""
    training_options = _sft_options_to_training_options(options)
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
        context = _build_sft_runtime_context(
            records=records,
            options=options,
            training_options=training_options,
            random_seed=random_seed,
            run_id=run_record.run_id,
            config_hash=config_hash,
            run_registry=run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = persist_sft_outputs(
            context=context,
            loop_result=loop_result,
            run_id=run_record.run_id,
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
            except CrucibleServeError:
                pass
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _build_sft_runtime_context(
    records: list[DataRecord],
    options: SftOptions,
    training_options: TrainingOptions,
    random_seed: int,
    run_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build runtime context using SFT-specific data and loss."""
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
    sft_examples = load_sft_examples(options.sft_data_path)
    sft_sequences = build_sft_sequences(
        examples=sft_examples,
        tokenizer=tokenizer,
        max_token_length=options.max_token_length,
        mask_prompt_tokens=options.mask_prompt_tokens,
    )
    if options.packing_enabled:
        sft_sequences = pack_sft_sequences(sft_sequences, options.max_token_length)
    if not sft_sequences:
        raise CrucibleSftError(
            "No trainable SFT sequences were generated. "
            "Check SFT data content and max token length."
        )
    random.Random(random_seed).shuffle(sft_sequences)
    train_batches, val_batches = build_sft_batches(sft_sequences, options)
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
    return TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision_runtime,
        loss_function=build_sft_loss_function(torch_module),
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device,
        run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )


def _sft_options_to_training_options(options: SftOptions) -> TrainingOptions:
    """Map SftOptions to TrainingOptions for reuse of shared components."""
    from core.training_types import options_to_training_options
    return options_to_training_options(options)


def _import_torch() -> Any:
    """Import torch dependency used by SFT training workflows."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "SFT training requires torch, but it is not installed. "
            "Install torch to run crucible sft."
        ) from error
    return torch
