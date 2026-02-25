"""DPO training runner for preference optimization workflows.

This module orchestrates DPO training: loads preference data, tokenizes
pairs, builds reference model, runs training loop, and persists artifacts.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.dpo_types import DpoOptions
from core.errors import ForgeDependencyError, ForgeDpoError
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.hf_model_loader import build_or_load_model
from serve.dpo_batch_processing import DpoContext, DpoLoopResult, run_dpo_loop
from serve.device_selection import resolve_execution_device
from serve.dpo_data_loader import load_dpo_examples
from serve.dpo_reference_model import create_reference_model, load_reference_model
from serve.dpo_tokenization import build_dpo_pairs
from serve.model_weights import load_initial_weights
from serve.training_artifact_contract import save_training_artifact_contract
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_model_weights,
    save_training_history,
    save_training_plot,
)
from serve.training_config_hash import compute_training_config_hash
from serve.training_hooks import load_training_hooks
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
from serve.training_reproducibility_bundle import save_reproducibility_bundle
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options


def run_dpo_training(
    records: list[DataRecord],
    options: DpoOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run a full DPO training workflow and persist run lifecycle metadata."""
    training_options = _dpo_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        dataset_version_id=dataset_version_id,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.initial_weights_path,
        config_hash=config_hash,
    )
    try:
        context = _build_dpo_context(
            records, options, training_options, random_seed,
        )
        run_registry.transition(run_record.run_id, "running")
        hooks = load_training_hooks(options.hooks_path)
        loop_result = run_dpo_loop(context, options)
        result = _persist_dpo_outputs(
            context=context,
            loop_result=loop_result,
            run_id=run_record.run_id,
            dataset_version_id=dataset_version_id,
            config_hash=config_hash,
            random_seed=random_seed,
        )
        run_registry.transition(
            run_id=run_record.run_id,
            next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(
            run_record.run_id, "failed", message=str(error)
        )
        raise


def _build_dpo_context(
    records: list[DataRecord],
    options: DpoOptions,
    training_options: TrainingOptions,
    random_seed: int,
) -> DpoContext:
    """Build DPO-specific runtime context."""
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
    dpo_examples = load_dpo_examples(options.dpo_data_path)
    dpo_pairs = build_dpo_pairs(
        examples=dpo_examples,
        tokenizer=tokenizer,
        max_length=options.max_token_length,
    )
    if not dpo_pairs:
        raise ForgeDpoError(
            "No trainable DPO pairs were generated. "
            "Check DPO data content and max token length."
        )
    random.Random(random_seed).shuffle(dpo_pairs)
    device = resolve_execution_device(torch_module)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=options.base_model,
        build_forge_model=lambda: load_training_model(torch_module, training_options, len(tokenizer.vocabulary)),
        device=device,
    )
    if not options.base_model:
        load_initial_weights(
            torch_module=torch_module, model=model,
            initial_weights_path=options.initial_weights_path, device=device,
        )
    if options.reference_model_path:
        ref_model = load_reference_model(
            torch_module, model, options.reference_model_path, device,
        )
    else:
        ref_model = create_reference_model(torch_module, model)
    return DpoContext(
        torch_module=torch_module,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dpo_pairs=dpo_pairs,
        output_dir=output_dir,
        device=device,
        training_options=training_options,
    )


def _persist_dpo_outputs(
    context: DpoContext,
    loop_result: DpoLoopResult,
    run_id: str,
    dataset_version_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist model/history outputs and return summary metadata."""
    model_path = save_model_weights(
        context.output_dir, context.torch_module, context.model,
    )
    config_path = save_training_config(
        context.output_dir, context.training_options,
    )
    tokenizer_path = save_tokenizer_vocabulary(
        context.output_dir, context.tokenizer,
    )
    history_path = save_training_history(
        context.output_dir, loop_result.epoch_metrics, [],
    )
    plot_path = _try_save_plot(
        context.output_dir, loop_result.epoch_metrics, [],
    )
    reproducibility_path = save_reproducibility_bundle(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        dataset_version_id=dataset_version_id,
        config_hash=config_hash,
        random_seed=random_seed,
        training_options=asdict(context.training_options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        run_id=run_id,
        artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        dataset_version_id=dataset_version_id,
        parent_model_path=context.training_options.initial_weights_path,
        config_hash=config_hash,
        result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(reproducibility_path),
    )
    return TrainingRunResult(
        model_path=base_result.model_path,
        history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        run_id=run_id,
        artifact_contract_path=str(contract_path),
    )


def _dpo_options_to_training_options(options: DpoOptions) -> TrainingOptions:
    """Map DpoOptions to TrainingOptions for reuse of shared components."""
    return TrainingOptions(
        dataset_name=options.dataset_name,
        output_dir=options.output_dir,
        version_id=options.version_id,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        batch_size=options.batch_size,
        max_token_length=options.max_token_length,
        validation_split=options.validation_split,
        precision_mode=options.precision_mode,
        optimizer_type=options.optimizer_type,
        weight_decay=options.weight_decay,
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        attention_heads=options.attention_heads,
        mlp_hidden_dim=options.mlp_hidden_dim,
        mlp_layers=options.mlp_layers,
        hooks_path=options.hooks_path,
        initial_weights_path=options.initial_weights_path,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        save_best_checkpoint=options.save_best_checkpoint,
        progress_log_interval_steps=options.progress_log_interval_steps,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
    )


def _import_torch() -> Any:
    """Import torch dependency used by DPO training workflows."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "DPO training requires torch, but it is not installed. "
            "Install torch to run forge dpo-train."
        ) from error
    return torch


def _try_save_plot(
    output_dir: Path,
    epoch_metrics: list[Any],
    batch_metrics: list[Any],
) -> Path | None:
    """Save training plot unless plotting dependency is unavailable."""
    try:
        return save_training_plot(output_dir, epoch_metrics, batch_metrics)
    except ForgeDependencyError:
        return None
