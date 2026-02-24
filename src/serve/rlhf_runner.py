"""RLHF training runner for policy optimization workflows.

This module orchestrates RLHF training: optionally trains a reward model,
then runs PPO to optimize the policy model using reward scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeRlhfError
from core.rlhf_types import RlhfOptions
from core.types import DataRecord, EpochMetric, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.model_weights import load_initial_weights
from serve.ppo_trainer import PpoEpochResult, run_ppo_epoch
from serve.reward_model import (
    build_reward_model_from_base,
    create_reference_policy,
    load_external_reward_model,
)
from serve.reward_model_trainer import train_reward_model
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
from serve.training_setup import fit_training_tokenizer


@dataclass
class _RlhfContext:
    """Runtime context for RLHF training."""

    torch_module: Any
    policy_model: Any
    reward_model: Any
    ref_model: Any
    tokenizer: Any
    output_dir: Path
    device: Any
    training_options: TrainingOptions


def run_rlhf_training(
    records: list[DataRecord],
    options: RlhfOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run a full RLHF training workflow and persist run lifecycle metadata."""
    training_options = _rlhf_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        dataset_version_id=dataset_version_id,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.policy_model_path,
        config_hash=config_hash,
    )
    try:
        context = _build_rlhf_context(records, options, training_options)
        run_registry.transition(run_record.run_id, "running")
        load_training_hooks(options.hooks_path)
        ppo_results = _run_ppo_training(context, options)
        epoch_metrics = _build_epoch_metrics(ppo_results)
        result = _persist_rlhf_outputs(
            context, epoch_metrics, run_record.run_id,
            dataset_version_id, config_hash, random_seed,
        )
        run_registry.transition(
            run_id=run_record.run_id, next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(
            run_record.run_id, "failed", message=str(error),
        )
        raise


def _build_rlhf_context(
    records: list[DataRecord],
    options: RlhfOptions,
    training_options: TrainingOptions,
) -> _RlhfContext:
    """Build RLHF runtime context with models and tokenizer."""
    torch_module = _import_torch()
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options)
    policy_model = load_training_model(
        torch_module, training_options, len(tokenizer.vocabulary),
    )
    device = resolve_execution_device(torch_module)
    policy_model = policy_model.to(device)
    load_initial_weights(
        torch_module=torch_module, model=policy_model,
        initial_weights_path=options.policy_model_path, device=device,
    )
    ref_model = create_reference_policy(torch_module, policy_model)
    reward_model = _resolve_reward_model(
        torch_module, policy_model, options, device,
    )
    return _RlhfContext(
        torch_module=torch_module, policy_model=policy_model,
        reward_model=reward_model, ref_model=ref_model,
        tokenizer=tokenizer, output_dir=output_dir,
        device=device, training_options=training_options,
    )


def _resolve_reward_model(
    torch_module: Any,
    policy_model: Any,
    options: RlhfOptions,
    device: Any,
) -> Any:
    """Resolve reward model: train from data or load external."""
    if options.reward_config.train_reward_model:
        return train_reward_model(
            torch_module, policy_model, options, device,
        )
    if options.reward_config.reward_model_path:
        reward_model = build_reward_model_from_base(
            torch_module, policy_model, options.hidden_dim,
        )
        return load_external_reward_model(
            torch_module, reward_model,
            options.reward_config.reward_model_path, device,
        )
    raise ForgeRlhfError(
        "RLHF training requires either --reward-model-path or "
        "--train-reward-model with --preference-data-path."
    )


def _run_ppo_training(
    context: _RlhfContext, options: RlhfOptions,
) -> list[PpoEpochResult]:
    """Execute PPO training loop across all epochs."""
    optimizer = context.torch_module.optim.Adam(
        context.policy_model.parameters(), lr=options.learning_rate,
    )
    prompts = _build_prompt_batch(context)
    results: list[PpoEpochResult] = []
    for epoch_idx in range(options.epochs):
        result = run_ppo_epoch(
            torch_module=context.torch_module,
            policy_model=context.policy_model,
            reward_model=context.reward_model,
            ref_model=context.ref_model,
            prompts=prompts, ppo_config=options.ppo_config,
            optimizer=optimizer, device=context.device,
            epoch=epoch_idx + 1,
        )
        results.append(result)
        print(
            f"RLHF epoch {result.epoch}/{options.epochs} "
            f"policy_loss={result.policy_loss:.4f} "
            f"mean_reward={result.mean_reward:.4f}"
        )
    return results


def _build_prompt_batch(context: _RlhfContext) -> Any:
    """Build a simple prompt batch from tokenizer vocabulary."""
    vocab_size = len(context.tokenizer.vocabulary)
    batch_size = min(4, vocab_size)
    return context.torch_module.randint(
        0, vocab_size, (batch_size, 16), device=context.device,
    )


def _build_epoch_metrics(ppo_results: list[PpoEpochResult]) -> list[EpochMetric]:
    """Convert PPO results to standard epoch metrics."""
    return [
        EpochMetric(epoch=r.epoch, train_loss=r.policy_loss, validation_loss=r.value_loss)
        for r in ppo_results
    ]


def _persist_rlhf_outputs(
    context: _RlhfContext,
    epoch_metrics: list[EpochMetric],
    run_id: str,
    dataset_version_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist model and history artifacts and return summary."""
    model_path = save_model_weights(
        context.output_dir, context.torch_module, context.policy_model,
    )
    config_path = save_training_config(context.output_dir, context.training_options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(context.output_dir, epoch_metrics, [])
    plot_path = _try_save_plot(context.output_dir, epoch_metrics, [])
    reproducibility_path = save_reproducibility_bundle(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        dataset_version_id=dataset_version_id,
        config_hash=config_hash, random_seed=random_seed,
        training_options=asdict(context.training_options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path), history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(epoch_metrics),
        run_id=run_id, artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.training_options.dataset_name,
        dataset_version_id=dataset_version_id,
        parent_model_path=context.training_options.initial_weights_path,
        config_hash=config_hash, result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(reproducibility_path),
    )
    return TrainingRunResult(
        model_path=base_result.model_path,
        history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _rlhf_options_to_training_options(options: RlhfOptions) -> TrainingOptions:
    """Map RlhfOptions to TrainingOptions for reuse of shared components."""
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
        hooks_path=options.hooks_path,
        initial_weights_path=options.initial_weights_path,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        save_best_checkpoint=options.save_best_checkpoint,
        progress_log_interval_steps=options.progress_log_interval_steps,
    )


def _import_torch() -> Any:
    """Import torch dependency used by RLHF training workflows."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "RLHF training requires torch, but it is not installed. "
            "Install torch to run forge rlhf-train."
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
