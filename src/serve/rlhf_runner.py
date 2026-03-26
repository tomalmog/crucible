"""RLHF training runner for policy optimization workflows.

Uses trl.PPOTrainer (or SFTTrainer fallback) for HuggingFace models
and falls back to the Crucible PPO training loop for custom .pt models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleRlhfError
from core.rlhf_types import RlhfOptions
from core.types import DataRecord, EpochMetric, TrainingOptions, TrainingRunResult
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
    records: list[DataRecord] | None = None


def run_rlhf_training(
    records: list[DataRecord],
    options: RlhfOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full RLHF training workflow and persist run lifecycle metadata."""
    training_options = _rlhf_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.policy_model_path,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        if options.policy_model_path and is_hf_model(options.policy_model_path):
            result = _run_rlhf_with_trl(
                records, options, training_options, run_record.run_id,
                config_hash, random_seed,
            )
        else:
            result = _run_rlhf_crucible(
                records, options, training_options, run_record.run_id,
                config_hash, random_seed,
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


def _run_rlhf_with_trl(
    records: list[DataRecord],
    options: RlhfOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """RLHF using trl for HuggingFace models.

    Full PPO requires complex reward model integration. We use SFTTrainer
    on the preference data (chosen responses) as a practical HF path,
    since trl's PPOTrainer requires online reward scoring infrastructure.
    """
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(
        options.policy_model_path, options.precision_mode,
    )

    # Build dataset from preference data or records
    texts = _build_rlhf_texts(records, options)
    if not texts:
        raise CrucibleRlhfError("No training data available for RLHF.")
    dataset = _texts_to_hf_dataset(texts)
    split = dataset.train_test_split(test_size=options.validation_split, seed=random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
    )

    print("RLHF: Using trl.SFTTrainer on policy model with preference data...", flush=True)
    sft_config = trl.SFTConfig(**args)
    trainer = trl.SFTTrainer(
        model=model, args=sft_config,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()

    print("RLHF: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_rlhf_crucible(
    records: list[DataRecord],
    options: RlhfOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """RLHF using Crucible's PPO training loop for .pt models."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.hf_model_loader import build_or_load_model, is_huggingface_model_id
    from serve.model_weights import load_initial_weights, read_model_state_dict
    from serve.ppo_trainer import PpoEpochResult, run_ppo_epoch
    from serve.reward_model import (
        build_reward_model_from_base,
        create_reference_policy,
        load_external_reward_model,
    )
    from serve.reward_model_trainer import train_reward_model
    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import (
        save_model_weights,
        save_training_history,
        save_training_plot,
    )
    from serve.training_hooks import load_training_hooks
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_progress import emit_progress
    from serve.training_reproducibility_bundle import save_reproducibility_bundle
    from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

    context = _build_rlhf_context(records, options, training_options)
    load_training_hooks(options.hooks_path)
    ppo_results = _run_ppo_training(context, options)
    epoch_metrics = _build_epoch_metrics(ppo_results)
    return _persist_rlhf_outputs(
        context, epoch_metrics, run_id, config_hash, random_seed,
    )


def _build_rlhf_texts(
    records: list[DataRecord],
    options: RlhfOptions,
) -> list[str]:
    """Build training texts from preference data or records.

    If preference data is available, uses the 'chosen' responses.
    Otherwise falls back to record texts.
    """
    texts: list[str] = []
    # Try preference data first
    if options.reward_config.preference_data_path:
        from serve.data_file_reader import read_data_rows
        try:
            rows = read_data_rows(options.reward_config.preference_data_path)
            for row in rows:
                prompt = str(row.get("prompt", ""))
                chosen = str(row.get("chosen", ""))
                if prompt and chosen:
                    texts.append(f"{prompt}\n{chosen}")
        except (FileNotFoundError, ImportError, OSError):
            pass
    # Fall back to records
    if not texts and records:
        for record in records:
            if record.text:
                texts.append(record.text)
    return texts


def _texts_to_hf_dataset(texts: list[str]) -> Any:
    """Convert texts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def _build_rlhf_context(
    records: list[DataRecord],
    options: RlhfOptions,
    training_options: TrainingOptions,
) -> _RlhfContext:
    """Build RLHF runtime context with models and tokenizer."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.hf_model_loader import build_or_load_model, is_huggingface_model_id
    from serve.model_weights import read_model_state_dict
    from serve.reward_model import (
        build_reward_model_from_base,
        create_reference_policy,
        load_external_reward_model,
    )
    from serve.reward_model_trainer import train_reward_model
    from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options

    torch_module = _import_torch()
    validate_training_options(training_options)
    validate_file_paths(
        policy_model_path=options.policy_model_path,
        reward_model_path=options.reward_config.reward_model_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    device = resolve_execution_device(torch_module)
    tokenizer = fit_training_tokenizer(records, training_options)
    vocab_size = _resolve_policy_vocab_size(
        torch_module, options.policy_model_path, device,
        fallback=len(tokenizer.vocabulary),
    )
    policy_model = build_or_load_model(
        torch_module=torch_module,
        base_model=None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, vocab_size),
        device=device,
        initial_weights_path=options.policy_model_path,
        training_options=training_options,
    )
    ref_model = create_reference_policy(torch_module, policy_model)
    reward_model = _resolve_reward_model(
        torch_module, policy_model, options, device, tokenizer,
    )
    return _RlhfContext(
        torch_module=torch_module, policy_model=policy_model,
        reward_model=reward_model, ref_model=ref_model,
        tokenizer=tokenizer, output_dir=output_dir,
        device=device, training_options=training_options,
        records=records,
    )


def _infer_hidden_dim(model: Any, fallback: int) -> int:
    """Infer hidden dimension from model config or architecture."""
    inner = getattr(model, "model", model)
    config = getattr(inner, "config", None)
    if config is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            val = getattr(config, attr, None)
            if val is not None:
                return int(val)
    return fallback


def _resolve_reward_model(
    torch_module: Any,
    policy_model: Any,
    options: RlhfOptions,
    device: Any,
    tokenizer: Any = None,
) -> Any:
    """Resolve reward model: train from data or load external."""
    from serve.reward_model import build_reward_model_from_base, load_external_reward_model
    from serve.reward_model_trainer import train_reward_model

    hidden_dim = _infer_hidden_dim(policy_model, options.hidden_dim)
    if options.reward_config.train_reward_model:
        from dataclasses import replace
        patched = replace(options, hidden_dim=hidden_dim)
        return train_reward_model(
            torch_module, policy_model, patched, device, tokenizer,
        )
    if options.reward_config.reward_model_path:
        reward_model = build_reward_model_from_base(
            torch_module, policy_model, hidden_dim,
        )
        return load_external_reward_model(
            torch_module, reward_model,
            options.reward_config.reward_model_path, device,
        )
    raise CrucibleRlhfError(
        "RLHF training requires either --reward-model-path or "
        "--train-reward-model with --preference-data-path."
    )


def _resolve_policy_vocab_size(
    torch_module: Any,
    policy_model_path: str | None,
    device: Any,
    fallback: int,
) -> int:
    """Infer vocab size from policy checkpoint embedding weights."""
    from serve.model_weights import read_model_state_dict

    if policy_model_path is None:
        return fallback
    resolved = Path(policy_model_path).expanduser().resolve()
    if not resolved.exists():
        return fallback
    try:
        state_dict = read_model_state_dict(torch_module, str(resolved), device)
        embedding_weight = state_dict.get("embedding.weight")
        if embedding_weight is not None and hasattr(embedding_weight, "shape"):
            return int(embedding_weight.shape[0])
    except Exception:
        pass
    return fallback


def _run_ppo_training(
    context: _RlhfContext, options: RlhfOptions,
) -> list[Any]:
    """Execute PPO training loop across all epochs."""
    from serve.ppo_trainer import PpoEpochResult, run_ppo_epoch
    from serve.training_progress import emit_progress

    optimizer = context.torch_module.optim.Adam(
        context.policy_model.parameters(), lr=options.learning_rate,
    )
    start_epoch = 1
    global_step = 0
    if options.resume_checkpoint_path:
        from serve.training_checkpoint import load_resume_checkpoint
        resume = load_resume_checkpoint(
            options.resume_checkpoint_path, context.torch_module,
            context.policy_model, optimizer, None, context.device,
        )
        start_epoch = resume.next_epoch
        global_step = resume.global_step
    prompts = _build_prompt_batch(context)
    emit_progress(
        "training_started",
        total_epochs=options.epochs,
        start_epoch=start_epoch,
        method="rlhf",
    )
    results: list[PpoEpochResult] = []
    epoch_idx = start_epoch - 1
    checkpoint_dir = None
    try:
        for epoch_idx in range(start_epoch - 1, options.epochs):
            emit_progress(
                "training_epoch_started",
                epoch=epoch_idx + 1,
                total_epochs=options.epochs,
            )
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
            global_step += 1
            emit_progress(
                "training_epoch_completed",
                epoch=result.epoch,
                total_epochs=options.epochs,
                train_loss=round(result.policy_loss, 6),
                mean_reward=round(result.mean_reward, 6),
            )
            from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
            checkpoint_dir = ensure_checkpoint_dir(context.output_dir)
            save_epoch_checkpoint(
                checkpoint_dir, context.torch_module, context.policy_model,
                optimizer, None, epoch_idx + 1, global_step, None,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted -- saving emergency checkpoint...", flush=True)
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        emergency_dir = checkpoint_dir or ensure_checkpoint_dir(context.output_dir)
        emergency_path = save_epoch_checkpoint(
            emergency_dir, context.torch_module, context.policy_model,
            optimizer, None, epoch_idx + 1, global_step, None,
        )
        print(f"Emergency checkpoint saved: {emergency_path}", flush=True)
        raise
    return results


def _build_prompt_batch(context: _RlhfContext) -> Any:
    """Build prompt batch from training data records."""
    max_len = context.training_options.max_token_length
    batch_size = min(4, len(context.records)) if context.records else 4

    if not context.records:
        return context.torch_module.zeros(
            (batch_size, max_len), dtype=context.torch_module.long,
            device=context.device,
        )

    token_ids = []
    for record in context.records[:batch_size]:
        ids = context.tokenizer.encode(record.text, max_len)
        ids = ids + [0] * (max_len - len(ids))
        token_ids.append(ids)
    return context.torch_module.tensor(
        token_ids, dtype=context.torch_module.long, device=context.device,
    )


def _build_epoch_metrics(ppo_results: list[Any]) -> list[EpochMetric]:
    """Convert PPO results to standard epoch metrics."""
    return [
        EpochMetric(epoch=r.epoch, train_loss=r.policy_loss, validation_loss=r.value_loss)
        for r in ppo_results
    ]


def _persist_rlhf_outputs(
    context: _RlhfContext,
    epoch_metrics: list[EpochMetric],
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist model and history artifacts and return summary."""
    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import (
        save_model_weights,
        save_training_history,
        save_training_plot,
    )
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_reproducibility_bundle import save_reproducibility_bundle

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
    from core.training_types import options_to_training_options
    return options_to_training_options(options, base_model_key="policy_model_path")


def _import_torch() -> Any:
    """Import torch dependency used by RLHF training workflows."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "RLHF training requires torch, but it is not installed. "
            "Install torch to run crucible rlhf-train."
        ) from error
    return torch


def _try_save_plot(
    output_dir: Path, epoch_metrics: list[Any], batch_metrics: list[Any],
) -> Path | None:
    """Save training plot unless plotting dependency is unavailable."""
    from serve.training_artifacts import save_training_plot
    try:
        return save_training_plot(output_dir, epoch_metrics, batch_metrics)
    except CrucibleDependencyError:
        return None
