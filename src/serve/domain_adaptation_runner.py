"""Domain adaptation runner for continued pretraining workflows.

This module orchestrates domain adaptation: loads a pretrained model,
continues training on domain-specific data, monitors drift on optional
reference data, and persists training artifacts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.domain_adaptation_types import DomainAdaptationOptions
from core.errors import ForgeDependencyError, ForgeServeError
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import build_or_load_model, is_huggingface_model_id
from serve.drift_detection import compute_perplexity
from serve.model_weights import load_initial_weights, read_model_state_dict
from serve.tokenization import build_sequence_batches, build_training_sequences, split_sequences
from serve.training_artifact_contract import save_training_artifact_contract
from serve.training_artifacts import (
    ensure_training_output_dir, save_model_weights,
    save_training_history, save_training_plot,
)
from serve.training_config_hash import compute_training_config_hash
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import TrainingLoopResult, run_training_loop
from serve.training_hooks import invoke_hook, load_training_hooks
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
from serve.training_optimization import build_training_optimization
from serve.training_precision import build_training_precision_runtime
from serve.training_reproducibility_bundle import save_reproducibility_bundle
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

logger = logging.getLogger(__name__)


def run_domain_adaptation(
    records: list[DataRecord],
    options: DomainAdaptationOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run domain adaptation with drift monitoring.

    Returns:
        Training run artifact summary.
    """
    training_options = _adaptation_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        dataset_version_id=dataset_version_id,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.base_model_path,
        config_hash=config_hash,
    )
    context: TrainingRuntimeContext | None = None
    try:
        context = _build_adaptation_context(
            records, options, training_options, random_seed,
            run_record.run_id, dataset_version_id, config_hash, run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = _persist_adaptation_outputs(
            context, loop_result, run_record.run_id,
            dataset_version_id, config_hash, random_seed,
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
            _invoke_error_hook(context, error)
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _build_adaptation_context(
    records: list[DataRecord],
    options: DomainAdaptationOptions,
    training_options: TrainingOptions,
    random_seed: int,
    run_id: str,
    dataset_version_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build runtime context for domain adaptation training."""
    torch_module = _import_torch()
    validate_training_options(training_options)
    validate_file_paths(
        base_model_path=options.base_model_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
        reference_data_path=options.reference_data_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options)
    sequences = build_training_sequences(records, tokenizer, options.max_token_length)
    if not sequences:
        raise ForgeServeError(
            "No trainable sequences generated from domain data. "
            "Check dataset content and max token length."
        )
    random.Random(random_seed).shuffle(sequences)
    train_seqs, val_seqs = split_sequences(sequences, options.validation_split)
    train_batches = build_sequence_batches(train_seqs, options.batch_size)
    val_batches = build_sequence_batches(val_seqs, options.batch_size)
    vocab_size = len(tokenizer.vocabulary)
    device = resolve_execution_device(torch_module)
    use_hf = options.base_model_path and is_huggingface_model_id(options.base_model_path)
    if options.base_model_path is not None and not use_hf:
        vocab_size = _infer_checkpoint_vocab_size(
            torch_module, options.base_model_path, device, vocab_size,
        )
        _pad_tokenizer_vocabulary(tokenizer, vocab_size)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=options.base_model_path if use_hf else None,
        build_forge_model=lambda: load_training_model(torch_module, training_options, vocab_size),
        device=device,
    )
    if not use_hf:
        load_initial_weights(
            torch_module=torch_module, model=model,
            initial_weights_path=options.base_model_path, device=device,
        )
    _run_drift_baseline(torch_module, model, options, tokenizer, device)
    precision = build_training_precision_runtime(
        torch_module=torch_module, requested_mode=options.precision_mode, device=device,
    )
    optimization = build_training_optimization(torch_module, model, training_options)
    hooks = load_training_hooks(options.hooks_path)
    return TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision,
        loss_function=torch_module.nn.CrossEntropyLoss(ignore_index=0),
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device, run_id=run_id,
        dataset_version_id=dataset_version_id, config_hash=config_hash,
        hooks=hooks, run_registry=run_registry,
    )


def _run_drift_baseline(
    torch_module: Any, model: Any,
    options: DomainAdaptationOptions, tokenizer: Any, device: Any,
) -> None:
    """Compute and log baseline perplexity on reference data."""
    if options.reference_data_path is None:
        return
    ref_seqs = _load_reference_sequences(
        options.reference_data_path, tokenizer, options.max_token_length,
    )
    if not ref_seqs:
        logger.warning("Reference data produced no sequences for drift check.")
        return
    baseline = compute_perplexity(torch_module, model, ref_seqs, device)
    logger.info("Baseline reference perplexity: %.4f", baseline)


def _load_reference_sequences(
    reference_path: str, tokenizer: Any, max_length: int,
) -> list[list[int]]:
    """Load reference text lines and tokenize for drift evaluation."""
    path = Path(reference_path).expanduser().resolve()
    if not path.exists():
        raise ForgeServeError(f"Reference data file not found at {path}.")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    sequences: list[list[int]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if len(tokens) >= 2:
            sequences.append(tokens)
    return sequences


def _persist_adaptation_outputs(
    context: TrainingRuntimeContext, loop_result: TrainingLoopResult,
    run_id: str, dataset_version_id: str, config_hash: str, random_seed: int,
) -> TrainingRunResult:
    """Persist adapted model and training artifacts."""
    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    config_path = save_training_config(context.output_dir, context.options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(
        context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
    )
    plot_path = _try_save_plot(
        context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
    )
    repro_path = save_reproducibility_bundle(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.options.dataset_name,
        dataset_version_id=dataset_version_id, config_hash=config_hash,
        random_seed=random_seed, training_options=asdict(context.options),
    )
    base = TrainingRunResult(
        model_path=str(model_path), history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        checkpoint_dir=str(loop_result.checkpoint_dir) if loop_result.checkpoint_dir else None,
        best_checkpoint_path=(
            str(loop_result.best_checkpoint_path) if loop_result.best_checkpoint_path else None
        ),
        resumed_from_checkpoint=loop_result.resumed_from_checkpoint,
        run_id=run_id, artifact_contract_path=None,
    )
    contract = save_training_artifact_contract(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.options.dataset_name,
        dataset_version_id=dataset_version_id,
        parent_model_path=context.options.initial_weights_path,
        config_hash=config_hash, result=base,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(repro_path),
    )
    return TrainingRunResult(
        model_path=base.model_path, history_path=base.history_path,
        plot_path=base.plot_path, epochs_completed=base.epochs_completed,
        checkpoint_dir=base.checkpoint_dir,
        best_checkpoint_path=base.best_checkpoint_path,
        resumed_from_checkpoint=base.resumed_from_checkpoint,
        run_id=run_id, artifact_contract_path=str(contract),
    )


def _infer_checkpoint_vocab_size(
    torch_module: Any, weights_path: str, device: Any, fallback: int,
) -> int:
    """Read embedding.weight from checkpoint to infer vocab size."""
    try:
        state = read_model_state_dict(torch_module, weights_path, device)
        embedding_weight = state.get("embedding.weight")
        if embedding_weight is not None and hasattr(embedding_weight, "shape"):
            return int(embedding_weight.shape[0])
    except ForgeServeError:
        pass
    return fallback


def _pad_tokenizer_vocabulary(
    tokenizer: Any, target_size: int,
) -> None:
    """Extend tokenizer vocabulary with placeholder tokens to match target size."""
    while len(tokenizer.vocabulary) < target_size:
        placeholder = f"<unused_{len(tokenizer.vocabulary)}>"
        tokenizer.vocabulary[placeholder] = len(tokenizer.vocabulary)


def _adaptation_to_training_options(options: DomainAdaptationOptions) -> TrainingOptions:
    """Map DomainAdaptationOptions to TrainingOptions."""
    return TrainingOptions(
        dataset_name=options.dataset_name, output_dir=options.output_dir,
        version_id=options.version_id, initial_weights_path=options.base_model_path,
        epochs=options.epochs, learning_rate=options.learning_rate,
        batch_size=options.batch_size, max_token_length=options.max_token_length,
        validation_split=options.validation_split, precision_mode=options.precision_mode,
        optimizer_type=options.optimizer_type, weight_decay=options.weight_decay,
        hidden_dim=options.hidden_dim, num_layers=options.num_layers,
        attention_heads=options.attention_heads,
        mlp_hidden_dim=options.mlp_hidden_dim,
        mlp_layers=options.mlp_layers,
        hooks_path=options.hooks_path,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        save_best_checkpoint=options.save_best_checkpoint,
        progress_log_interval_steps=options.progress_log_interval_steps,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
    )


def _import_torch() -> Any:
    """Import torch dependency for domain adaptation."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Domain adaptation requires torch, but it is not installed. "
            "Install torch to run forge domain-adapt."
        ) from error
    return torch


def _try_save_plot(
    output_dir: Path, epoch_metrics: list[Any], batch_metrics: list[Any],
) -> Path | None:
    """Save training plot unless plotting dependency is unavailable."""
    try:
        return save_training_plot(output_dir, epoch_metrics, batch_metrics)
    except ForgeDependencyError:
        return None


def _invoke_error_hook(context: TrainingRuntimeContext, error: Exception) -> None:
    """Invoke error hook without replacing the original failure."""
    try:
        invoke_hook("on_run_error", context.hooks.on_run_error, context, str(error))
    except ForgeServeError:
        return
