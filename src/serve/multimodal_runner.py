"""Multimodal training runner for vision-language fine-tuning.

Keeps the existing Crucible implementation for both HF and .pt paths
since multimodal training is complex and trl's SFTTrainer multimodal
support requires model-specific configuration.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleServeError
from core.multimodal_types import MultimodalOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import build_or_load_model
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


def run_multimodal_training(
    records: list[DataRecord], options: MultimodalOptions,
    random_seed: int, data_root: Path,
) -> TrainingRunResult:
    """Run multimodal fine-tuning workflow.

    Multimodal training uses the Crucible training loop for both HF and
    .pt models. trl's SFTTrainer can handle multimodal if the model
    supports it, but the configuration is model-specific and complex.
    """
    training_options = _mm_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.initial_weights_path or options.base_model,
        config_hash=config_hash,
    )
    context: TrainingRuntimeContext | None = None
    try:
        context = _build_mm_context(
            records, options, training_options, random_seed,
            run_record.run_id, config_hash, run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = _persist_mm_outputs(context, loop_result, run_record.run_id, config_hash, random_seed)
        invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
        run_registry.transition(run_id=run_record.run_id, next_state="completed",
            artifact_contract_path=result.artifact_contract_path, model_path=result.model_path)
        return result
    except Exception as error:
        if context is not None:
            try:
                invoke_hook("on_run_error", context.hooks.on_run_error, context, str(error))
            except CrucibleServeError:
                pass
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _build_mm_context(
    records: list[DataRecord],
    options: MultimodalOptions,
    training_options: TrainingOptions,
    random_seed: int,
    run_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build runtime context for multimodal training."""
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
    data = _load_mm_data(options.multimodal_data_path)
    if not data:
        raise CrucibleServeError("No multimodal data loaded.")
    random.Random(random_seed).shuffle(data)
    train_batches, val_batches = _build_mm_batches(data, tokenizer, options)
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
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=0)
    return TrainingRuntimeContext(
        torch_module=torch_module, model=model, optimizer=optimization.optimizer,
        scheduler=optimization.scheduler,
        precision_runtime=precision_runtime, loss_function=loss_fn,
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options, output_dir=output_dir,
        device=device, run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )


def _load_mm_data(data_path: str) -> list[dict[str, object]]:
    """Load multimodal data from JSONL or Parquet."""
    from serve.data_file_reader import read_data_rows

    try:
        return read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise CrucibleServeError(str(exc)) from exc


def _build_mm_batches(
    data: list[dict[str, object]], tokenizer: Any, options: MultimodalOptions,
) -> tuple[list[Any], list[Any]]:
    """Build train/val token batches from multimodal data."""
    from serve.tokenization import SequenceBatch

    split_idx = max(1, int(len(data) * (1.0 - options.validation_split)))

    def to_batches(examples: list[dict[str, object]]) -> list[Any]:
        batches = []
        for i in range(0, len(examples), options.batch_size):
            batch = examples[i:i + options.batch_size]
            token_ids = []
            for ex in batch:
                text = ex.get("text", ex.get("caption", ""))
                ids = tokenizer.encode(text, options.max_token_length)
                padded = ids + [0] * (options.max_token_length - len(ids))
                token_ids.append(padded)
            batches.append(SequenceBatch(
                inputs=[seq[:-1] for seq in token_ids],
                targets=[seq[1:] for seq in token_ids],
            ))
        return batches

    train = to_batches(data[:split_idx])
    val = to_batches(data[split_idx:]) if split_idx < len(data) else []
    return train, val


def _persist_mm_outputs(
    context: TrainingRuntimeContext, loop_result: Any,
    run_id: str, config_hash: str, random_seed: int,
) -> TrainingRunResult:
    """Persist multimodal training outputs."""
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
        reproducibility_bundle_path=None,
    )
    return TrainingRunResult(
        model_path=base_result.model_path, history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _mm_options_to_training_options(options: MultimodalOptions) -> TrainingOptions:
    """Map MultimodalOptions to TrainingOptions."""
    from core.training_types import options_to_training_options
    return options_to_training_options(options)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError("Multimodal training requires torch.") from error
    return torch
