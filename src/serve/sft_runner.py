"""SFT training runner for supervised fine-tuning workflows.

Uses trl.SFTTrainer for HuggingFace models and falls back to the
Crucible training loop for custom .pt models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleSftError
from core.sft_types import SftOptions
from core.types import DataRecord, TrainingRunResult
from core.training_types import options_to_training_options
from serve.sft_data_loader import load_sft_examples
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


def run_sft_training(
    records: list[DataRecord],
    options: SftOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full SFT training workflow and persist run lifecycle metadata."""
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
            result = _run_sft_with_trl(options, training_options, run_record.run_id, config_hash, random_seed)
        else:
            result = _run_sft_crucible(records, options, training_options, run_record.run_id, config_hash, random_seed, data_root)
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


def _run_sft_with_trl(
    options: SftOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """SFT using trl.SFTTrainer for HuggingFace models."""
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(options.base_model, options.precision_mode)

    sft_examples = load_sft_examples(options.sft_data_path)
    if not sft_examples:
        raise CrucibleSftError("No SFT examples found.")
    dataset = _examples_to_hf_dataset(sft_examples)
    train_dataset, eval_dataset = split_dataset(dataset, options.validation_split, random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
        has_eval=eval_dataset is not None,
    )
    sft_config = trl.SFTConfig(**args)

    print("SFT: Starting training...", flush=True)
    trainer = trl.SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=options.resume_checkpoint_path)

    print("SFT: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_sft_crucible(
    records: list[DataRecord],
    options: SftOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """SFT using Crucible's custom training loop for .pt models."""
    import random as random_module
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.hf_model_loader import build_or_load_model
    from serve.sft_batch_builder import build_sft_batches, persist_sft_outputs
    from serve.sft_loss import build_sft_loss_function
    from serve.sft_tokenization import build_sft_sequences, pack_sft_sequences
    from serve.training_context import TrainingRuntimeContext
    from serve.training_execution import run_training_loop
    from serve.training_hooks import load_training_hooks
    from serve.training_optimization import build_training_optimization
    from serve.training_precision import build_training_precision_runtime
    from serve.training_setup import fit_training_tokenizer

    torch_module = _import_torch()
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options, base_model=options.base_model)
    sft_examples = load_sft_examples(options.sft_data_path)
    sft_sequences = build_sft_sequences(
        examples=sft_examples, tokenizer=tokenizer,
        max_token_length=options.max_token_length, mask_prompt_tokens=options.mask_prompt_tokens,
    )
    if options.packing_enabled:
        sft_sequences = pack_sft_sequences(sft_sequences, options.max_token_length)
    if not sft_sequences:
        raise CrucibleSftError("No trainable SFT sequences were generated.")
    random_module.Random(random_seed).shuffle(sft_sequences)
    train_batches, val_batches = build_sft_batches(sft_sequences, options)
    device = resolve_execution_device(torch_module)
    model = build_or_load_model(
        torch_module=torch_module, base_model=None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, len(tokenizer.vocabulary)),
        device=device, initial_weights_path=options.initial_weights_path, training_options=training_options,
    )
    precision_runtime = build_training_precision_runtime(torch_module=torch_module, requested_mode=options.precision_mode, device=device)
    optimization = build_training_optimization(torch_module, model, training_options)
    hooks = load_training_hooks(options.hooks_path)
    run_registry = TrainingRunRegistry(data_root)
    context = TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision_runtime, loss_function=build_sft_loss_function(torch_module),
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device, run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )
    loop_result = run_training_loop(context)
    return persist_sft_outputs(context=context, loop_result=loop_result, run_id=run_id, config_hash=config_hash, random_seed=random_seed)


def _examples_to_hf_dataset(examples: list[Any]) -> Any:
    from datasets import Dataset
    texts = []
    for ex in examples:
        text = f"{ex.system_prompt}\n\n" if ex.system_prompt else ""
        text += f"{ex.prompt}\n{ex.response}"
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def _import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as error:
        raise CrucibleDependencyError("SFT training requires torch.") from error
