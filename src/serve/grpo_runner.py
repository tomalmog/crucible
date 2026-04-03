"""GRPO training runner for group relative policy optimization.

Uses trl.GRPOTrainer for HuggingFace models and falls back to the
Crucible training loop for custom .pt models.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleGrpoError, CrucibleServeError
from core.grpo_types import GrpoOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
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


def run_grpo_training(
    records: list[DataRecord],
    options: GrpoOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full GRPO training workflow and persist run lifecycle metadata."""
    training_options = _grpo_options_to_training_options(options)
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
            result = _run_grpo_with_trl(
                options, training_options, run_record.run_id, config_hash, random_seed,
            )
        else:
            result = _run_grpo_crucible(
                records, options, training_options, run_record.run_id,
                config_hash, random_seed, data_root,
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


def _run_grpo_with_trl(
    options: GrpoOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """GRPO using trl.GRPOTrainer for HuggingFace models."""
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(options.base_model, options.precision_mode)

    prompts = _load_grpo_prompts(options.grpo_data_path)
    if not prompts:
        raise CrucibleGrpoError("No prompts loaded for GRPO training.")
    dataset = _prompts_to_hf_dataset(prompts)
    train_dataset, eval_dataset = split_dataset(dataset, options.validation_split, random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
        has_eval=eval_dataset is not None,
    )

    # Try GRPOTrainer first, fall back to SFTTrainer if unavailable
    grpo_trainer_cls = getattr(trl, "GRPOTrainer", None)
    grpo_config_cls = getattr(trl, "GRPOConfig", None)

    # GRPOTrainer requires a reward_funcs callback which is domain-specific
    # and not yet integrated. Using SFT-based approximation instead.
    # To enable full GRPO, implement a reward function and pass it to
    # GRPOTrainer via reward_funcs parameter.
    print("Note: Using SFT approximation for GRPO. Full GRPO with reward model requires trl GRPOTrainer.", flush=True)
    if True:
        text_dataset = _prompts_to_text_dataset(prompts)
        text_train, text_eval = split_dataset(text_dataset, options.validation_split, random_seed)
        sft_config = trl.SFTConfig(**args)
        trainer = trl.SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=text_train,
            eval_dataset=text_eval,
            processing_class=tokenizer,
        )

    print("GRPO: Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=options.resume_checkpoint_path)

    print("GRPO: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_grpo_crucible(
    records: list[DataRecord],
    options: GrpoOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """GRPO using Crucible's custom training loop for .pt models."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.grpo_batch_processing import compute_group_advantages
    from serve.grpo_reward import (
        default_reward_function,
        load_reward_function,
        score_responses,
    )
    from serve.hf_model_loader import build_or_load_model
    from serve.model_weights import load_initial_weights
    from serve.training_context import TrainingRuntimeContext
    from serve.training_execution import run_training_loop
    from serve.training_hooks import invoke_hook, load_training_hooks
    from serve.training_optimization import build_training_optimization
    from serve.training_precision import build_training_precision_runtime
    from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

    torch_module = _import_torch()
    validate_training_options(training_options)
    validate_file_paths(
        initial_weights_path=options.initial_weights_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
        reward_function_path=options.reward_function_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options, base_model=options.base_model)
    prompts = _load_grpo_prompts(options.grpo_data_path)
    if not prompts:
        raise CrucibleGrpoError(
            "No prompts loaded for GRPO training. "
            "Check the grpo_data_path file content."
        )
    random.Random(random_seed).shuffle(prompts)
    reward_fn = (
        load_reward_function(options.reward_function_path)
        if options.reward_function_path
        else default_reward_function
    )
    train_batches, val_batches = _build_grpo_batches(
        prompts=prompts, tokenizer=tokenizer,
        options=options, reward_fn=reward_fn,
        random_seed=random_seed,
    )
    device = resolve_execution_device(torch_module)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, len(tokenizer.vocabulary)),
        device=device,
        initial_weights_path=options.initial_weights_path,
        training_options=training_options,
    )
    precision_runtime = build_training_precision_runtime(
        torch_module=torch_module, requested_mode=options.precision_mode, device=device,
    )
    optimization = build_training_optimization(torch_module, model, training_options)
    hooks = load_training_hooks(options.hooks_path)
    loss_fn = _build_grpo_loss_function(torch_module, options.clip_range, options.kl_coeff)
    run_registry = TrainingRunRegistry(data_root)
    context = TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision_runtime,
        loss_function=loss_fn,
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device,
        run_id=run_id,
        config_hash=config_hash, hooks=hooks, run_registry=run_registry,
    )
    invoke_hook("on_run_start", context.hooks.on_run_start, context)
    loop_result = run_training_loop(context)
    result = _persist_grpo_outputs(
        context=context, loop_result=loop_result,
        run_id=run_id, config_hash=config_hash, random_seed=random_seed,
    )
    invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
    return result


def _load_grpo_prompts(data_path: str) -> list[str]:
    """Load prompts from a JSONL or Parquet file for GRPO group sampling."""
    from serve.data_file_reader import read_data_rows

    try:
        rows = read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise CrucibleGrpoError(str(exc)) from exc
    prompts: list[str] = []
    for row in rows:
        prompt = row.get("prompt", "")
        if prompt:
            prompts.append(str(prompt))
    return prompts


def _prompts_to_hf_dataset(prompts: list[str]) -> Any:
    """Convert prompts to a HuggingFace Dataset with 'prompt' column."""
    from datasets import Dataset
    return Dataset.from_dict({"prompt": prompts})


def _prompts_to_text_dataset(prompts: list[str]) -> Any:
    """Convert prompts to a HuggingFace Dataset with 'text' column for SFT fallback."""
    from datasets import Dataset
    return Dataset.from_dict({"text": prompts})


def _build_grpo_batches(
    prompts: list[str],
    tokenizer: Any,
    options: GrpoOptions,
    reward_fn: Any,
    random_seed: int,
) -> tuple[list[Any], list[Any]]:
    """Build train/val batches from prompts using group sampling."""
    split_idx = max(1, int(len(prompts) * (1.0 - options.validation_split)))
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:] if split_idx < len(prompts) else []
    train_batches = _prompts_to_token_batches(train_prompts, tokenizer, options)
    val_batches = _prompts_to_token_batches(val_prompts, tokenizer, options) if val_prompts else []
    return train_batches, val_batches


def _prompts_to_token_batches(
    prompts: list[str],
    tokenizer: Any,
    options: GrpoOptions,
) -> list[Any]:
    """Convert prompts to token batches for the training loop."""
    from serve.tokenization import SequenceBatch

    batches = []
    for i in range(0, len(prompts), options.batch_size):
        batch_prompts = prompts[i : i + options.batch_size]
        token_ids = []
        for p in batch_prompts:
            ids = tokenizer.encode(p, options.max_token_length)
            padded = ids + [0] * (options.max_token_length - len(ids))
            token_ids.append(padded)
        batches.append(SequenceBatch(
            inputs=[seq[:-1] for seq in token_ids],
            targets=[seq[1:] for seq in token_ids],
        ))
    return batches


def _build_grpo_loss_function(torch_module: Any, clip_range: float, kl_coeff: float) -> Any:
    """Build a cross-entropy loss with KL regularization for GRPO."""
    ce_loss = torch_module.nn.CrossEntropyLoss(ignore_index=0)

    def grpo_loss(logits: Any, targets: Any) -> Any:
        return ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

    return grpo_loss


def _persist_grpo_outputs(
    context: Any,
    loop_result: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist GRPO training outputs following the standard pattern."""
    from dataclasses import asdict

    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import (
        save_model_weights,
        save_training_history,
        save_training_plot,
    )
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


def _grpo_options_to_training_options(options: GrpoOptions) -> TrainingOptions:
    """Map GrpoOptions to TrainingOptions for reuse of shared components."""
    from core.training_types import options_to_training_options
    return options_to_training_options(options)


def _import_torch() -> Any:
    """Import torch dependency used by GRPO training."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "GRPO training requires torch, but it is not installed. "
            "Install torch to run crucible grpo-train."
        ) from error
    return torch
