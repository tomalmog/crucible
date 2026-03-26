"""ORPO training runner for odds ratio preference optimization.

Uses trl.ORPOTrainer for HuggingFace models and falls back to the
Crucible training loop for custom .pt models.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleOrpoError, CrucibleServeError
from core.orpo_types import OrpoOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from core.training_types import options_to_training_options
from serve.training_artifacts import ensure_training_output_dir
from serve.training_config_hash import compute_training_config_hash
from serve.training_run_registry import TrainingRunRegistry
from serve.trl_training_base import (
    build_base_training_args,
    is_hf_model,
    load_hf_model_and_tokenizer,
    save_trl_outputs,
    _import_trl,
)


def run_orpo_training(
    records: list[DataRecord],
    options: OrpoOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full ORPO training workflow and persist run lifecycle metadata."""
    training_options = options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.initial_weights_path or options.base_model,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        if options.base_model and is_hf_model(options.base_model):
            result = _run_orpo_with_trl(
                options, training_options, run_record.run_id, config_hash, random_seed,
            )
        else:
            result = _run_orpo_crucible(
                records, options, training_options, run_record.run_id, config_hash, random_seed, data_root,
            )
        run_registry.transition(
            run_id=run_record.run_id,
            next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _run_orpo_with_trl(
    options: OrpoOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """ORPO using trl.ORPOTrainer for HuggingFace models."""
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(options.base_model, options.precision_mode)

    orpo_data = _load_orpo_data(options.orpo_data_path)
    if not orpo_data:
        raise CrucibleOrpoError("No ORPO examples found.")
    dataset = _orpo_data_to_hf_dataset(orpo_data)
    split = dataset.train_test_split(test_size=options.validation_split, seed=random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
    )
    args["beta"] = options.beta
    # trl >= 0.30 has ORPOConfig/ORPOTrainer; older versions fall back to DPO
    config_cls = getattr(trl, "ORPOConfig", None) or trl.DPOConfig
    trainer_cls = getattr(trl, "ORPOTrainer", None) or trl.DPOTrainer
    orpo_config = config_cls(**args)

    print("ORPO: Starting training...", flush=True)
    trainer = trainer_cls(
        model=model, args=orpo_config,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()

    print("ORPO: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_orpo_crucible(
    records: list[DataRecord],
    options: OrpoOptions,
    training_options: TrainingOptions,
    run_id: str,
    config_hash: str,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """ORPO using Crucible's custom training loop for .pt models."""
    from dataclasses import asdict

    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.hf_model_loader import build_or_load_model
    from serve.orpo_loss import build_orpo_loss_function
    from serve.tokenization import SequenceBatch
    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import save_model_weights, save_training_history, save_training_plot
    from serve.training_context import TrainingRuntimeContext
    from serve.training_execution import run_training_loop
    from serve.training_hooks import invoke_hook, load_training_hooks
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_optimization import build_training_optimization
    from serve.training_precision import build_training_precision_runtime
    from serve.training_reproducibility_bundle import save_reproducibility_bundle
    from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

    torch_module = _import_torch()
    run_registry = TrainingRunRegistry(data_root)
    validate_training_options(training_options)
    validate_file_paths(
        initial_weights_path=options.initial_weights_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options, base_model=options.base_model)
    orpo_data = _load_orpo_data(options.orpo_data_path)
    if not orpo_data:
        raise CrucibleOrpoError(
            "No ORPO preference data loaded. Check orpo_data_path file content."
        )
    random.Random(random_seed).shuffle(orpo_data)
    train_batches, val_batches = _build_orpo_batches(
        data=orpo_data, tokenizer=tokenizer, options=options,
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
    loss_fn = build_orpo_loss_function(torch_module, options.lambda_orpo, options.beta)
    context = TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision_runtime, loss_function=loss_fn,
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device,
        run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )
    invoke_hook("on_run_start", context.hooks.on_run_start, context)
    loop_result = run_training_loop(context)

    # Persist outputs
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
    invoke_hook("on_run_end", context.hooks.on_run_end, context, base_result)
    return TrainingRunResult(
        model_path=base_result.model_path, history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _load_orpo_data(data_path: str) -> list[dict[str, str]]:
    """Load ORPO preference pairs from JSONL or Parquet file."""
    from serve.data_file_reader import read_data_rows

    try:
        rows = read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise CrucibleOrpoError(str(exc)) from exc
    data: list[dict[str, str]] = []
    for row in rows:
        if "prompt" in row and "chosen" in row:
            data.append({k: str(v) for k, v in row.items()})
    return data


def _build_orpo_batches(
    data: list[dict[str, str]],
    tokenizer: Any,
    options: OrpoOptions,
) -> tuple[list[Any], list[Any]]:
    """Build train/val batches from ORPO preference pairs."""
    from serve.tokenization import SequenceBatch

    split_idx = max(1, int(len(data) * (1.0 - options.validation_split)))
    train_data = data[:split_idx]
    val_data = data[split_idx:] if split_idx < len(data) else []

    def to_batches(examples: list[dict[str, str]]) -> list[Any]:
        batches = []
        for i in range(0, len(examples), options.batch_size):
            batch = examples[i : i + options.batch_size]
            all_inputs = []
            all_targets = []
            for ex in batch:
                prompt = ex.get("prompt", "")
                chosen_text = prompt + " " + ex.get("chosen", "")
                rejected_text = prompt + " " + ex.get("rejected", "")
                chosen_ids = tokenizer.encode(chosen_text, options.max_token_length)
                rejected_ids = tokenizer.encode(rejected_text, options.max_token_length)
                chosen_padded = chosen_ids + [0] * (options.max_token_length - len(chosen_ids))
                rejected_padded = rejected_ids + [0] * (options.max_token_length - len(rejected_ids))
                all_inputs.append(chosen_padded)
                all_inputs.append(rejected_padded)
                all_targets.append(list(chosen_padded))
                all_targets.append(list(rejected_padded))
            batches.append(SequenceBatch(inputs=all_inputs, targets=all_targets))
        return batches

    return to_batches(train_data), to_batches(val_data)


def _orpo_data_to_hf_dataset(data: list[dict[str, str]]) -> Any:
    """Convert ORPO data dicts to HF Dataset with prompt/chosen/rejected columns."""
    from datasets import Dataset
    return Dataset.from_dict({
        "prompt": [d["prompt"] for d in data],
        "chosen": [d["chosen"] for d in data],
        "rejected": [d.get("rejected", "") for d in data],
    })


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "ORPO training requires torch. Install torch to run crucible orpo-train."
        ) from error
    return torch
