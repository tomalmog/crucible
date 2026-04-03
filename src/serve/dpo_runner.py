"""DPO training runner for preference optimization workflows.

Uses trl.DPOTrainer for HuggingFace models and falls back to the
Crucible training loop for custom .pt models.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.dpo_types import DpoOptions
from core.errors import CrucibleDependencyError, CrucibleDpoError
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from core.training_types import options_to_training_options
from serve.dpo_data_loader import load_dpo_examples
from serve.training_artifacts import ensure_training_output_dir
from serve.training_config_hash import compute_training_config_hash
from serve.training_run_registry import TrainingRunRegistry
from serve.trl_training_base import (
    build_base_training_args,
    is_hf_model,
    load_hf_model_and_tokenizer,
    save_trl_outputs,
    split_dataset,
    _import_trl,
)


def run_dpo_training(
    records: list[DataRecord],
    options: DpoOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full DPO training workflow and persist run lifecycle metadata."""
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
            result = _run_dpo_with_trl(
                options, training_options, run_record.run_id, config_hash, random_seed,
            )
        else:
            result = _run_dpo_crucible(
                records, options, training_options, run_record.run_id, config_hash, random_seed,
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


def _run_dpo_with_trl(
    options: DpoOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """DPO using trl.DPOTrainer for HuggingFace models."""
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(options.base_model, options.precision_mode)

    dpo_examples = load_dpo_examples(options.dpo_data_path)
    if not dpo_examples:
        raise CrucibleDpoError("No DPO examples found.")
    dataset = _dpo_examples_to_hf_dataset(dpo_examples)
    train_dataset, eval_dataset = split_dataset(dataset, options.validation_split, random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
        has_eval=eval_dataset is not None,
    )
    args["beta"] = options.beta
    dpo_config = trl.DPOConfig(**args)

    print("DPO: Starting training...", flush=True)
    trainer = trl.DPOTrainer(
        model=model, args=dpo_config,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=options.resume_checkpoint_path)

    print("DPO: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_dpo_crucible(
    records: list[DataRecord],
    options: DpoOptions,
    training_options: TrainingOptions,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """DPO using Crucible's custom training loop for .pt models."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.dpo_batch_processing import DpoContext, run_dpo_loop
    from serve.dpo_reference_model import create_reference_model, load_reference_model
    from serve.dpo_tokenization import build_dpo_pairs
    from serve.hf_model_loader import build_or_load_model
    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import (
        save_model_weights,
        save_training_history,
        save_training_plot,
    )
    from serve.training_hooks import load_training_hooks
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_reproducibility_bundle import save_reproducibility_bundle
    from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

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
        raise CrucibleDpoError(
            "No trainable DPO pairs were generated. "
            "Check DPO data content and max token length."
        )
    random.Random(random_seed).shuffle(dpo_pairs)
    # Split into train/val sets
    val_size = max(1, int(len(dpo_pairs) * options.validation_split))
    train_pairs = dpo_pairs[val_size:]
    val_pairs = dpo_pairs[:val_size]
    device = resolve_execution_device(torch_module)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=options.base_model,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, len(tokenizer.vocabulary)),
        device=device,
        initial_weights_path=options.initial_weights_path if not options.base_model else None,
        training_options=training_options,
    )
    if options.reference_model_path:
        ref_model = load_reference_model(
            torch_module, model, options.reference_model_path, device,
        )
    else:
        ref_model = create_reference_model(torch_module, model)
    context = DpoContext(
        torch_module=torch_module,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dpo_pairs=train_pairs,
        output_dir=output_dir,
        device=device,
        training_options=training_options,
        val_pairs=val_pairs,
    )
    hooks = load_training_hooks(options.hooks_path)
    loop_result = run_dpo_loop(context, options)

    # Persist outputs
    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    config_path = save_training_config(context.output_dir, context.training_options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(context.output_dir, loop_result.epoch_metrics, [])
    try:
        plot_path = save_training_plot(context.output_dir, loop_result.epoch_metrics, [])
    except CrucibleDependencyError:
        plot_path = None
    reproducibility_path = save_reproducibility_bundle(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        config_hash=config_hash, random_seed=random_seed,
        training_options=asdict(context.training_options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path), history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        run_id=run_id, artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        parent_model_path=context.training_options.initial_weights_path,
        config_hash=config_hash, result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(reproducibility_path),
    )
    return TrainingRunResult(
        model_path=base_result.model_path, history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _dpo_examples_to_hf_dataset(examples: list[Any]) -> Any:
    """Convert DpoExample list to HF Dataset with prompt/chosen/rejected columns."""
    from datasets import Dataset
    return Dataset.from_dict({
        "prompt": [ex.prompt for ex in examples],
        "chosen": [ex.chosen for ex in examples],
        "rejected": [ex.rejected for ex in examples],
    })


def _import_torch() -> Any:
    """Import torch dependency used by DPO training workflows."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "DPO training requires torch, but it is not installed. "
            "Install torch to run crucible dpo-train."
        ) from error
    return torch
