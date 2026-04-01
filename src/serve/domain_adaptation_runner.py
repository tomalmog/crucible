"""Domain adaptation runner for continued pretraining workflows.

Uses trl.SFTTrainer for HuggingFace models and falls back to the
Crucible training loop for custom .pt models. Domain adaptation is
essentially continued pretraining on domain-specific data.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.domain_adaptation_types import DomainAdaptationOptions
from core.errors import CrucibleDependencyError, CrucibleServeError
from core.types import DataRecord, TrainingOptions, TrainingRunResult
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

logger = logging.getLogger(__name__)


def run_domain_adaptation(
    records: list[DataRecord],
    options: DomainAdaptationOptions,
    random_seed: int,
    data_root: Path,
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
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.base_model_path,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        if options.base_model_path and is_hf_model(options.base_model_path):
            result = _run_adaptation_with_trl(
                records, options, training_options, run_record.run_id,
                config_hash, random_seed,
            )
        else:
            result = _run_adaptation_crucible(
                records, options, training_options, random_seed,
                run_record.run_id, config_hash, data_root,
            )
        run_registry.transition(
            run_id=run_record.run_id, next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def _run_adaptation_with_trl(
    records: list[DataRecord],
    options: DomainAdaptationOptions,
    training_options: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Domain adaptation using trl.SFTTrainer for HuggingFace models.

    Continued pretraining on domain text is a natural fit for SFTTrainer
    with plain text data.
    """
    trl = _import_trl()
    output_dir = ensure_training_output_dir(options.output_dir)
    model, tokenizer = load_hf_model_and_tokenizer(
        options.base_model_path, options.precision_mode,
    )

    # Build text dataset from records
    texts = [record.text for record in records if record.text]
    if not texts:
        raise CrucibleServeError(
            "No trainable text found in domain data. "
            "Check dataset content."
        )
    dataset = _texts_to_hf_dataset(texts)
    split = dataset.train_test_split(test_size=options.validation_split, seed=random_seed)

    args = build_base_training_args(
        output_dir=output_dir, epochs=options.epochs, batch_size=options.batch_size,
        learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        precision_mode=options.precision_mode, log_steps=options.progress_log_interval_steps,
        seed=random_seed, max_length=options.max_token_length,
    )

    print("Domain Adaptation: Starting continued pretraining with trl.SFTTrainer...", flush=True)
    sft_config = trl.SFTConfig(**args)
    trainer = trl.SFTTrainer(
        model=model, args=sft_config,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=options.resume_checkpoint_path)

    print("Domain Adaptation: Saving model...", flush=True)
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _run_adaptation_crucible(
    records: list[DataRecord],
    options: DomainAdaptationOptions,
    training_options: Any,
    random_seed: int,
    run_id: str,
    config_hash: str,
    data_root: Path,
) -> TrainingRunResult:
    """Domain adaptation using Crucible's custom training loop for .pt models."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.drift_detection import compute_perplexity
    from serve.hf_model_loader import build_or_load_model, is_huggingface_model_id
    from serve.model_weights import read_model_state_dict
    from serve.tokenization import build_sequence_batches, build_training_sequences, split_sequences
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
        raise CrucibleServeError(
            "No trainable sequences generated from domain data. "
            "Check dataset content and max token length."
        )
    random.Random(random_seed).shuffle(sequences)
    train_seqs, val_seqs = split_sequences(sequences, options.validation_split)
    train_batches = build_sequence_batches(train_seqs, options.batch_size)
    val_batches = build_sequence_batches(val_seqs, options.batch_size)
    vocab_size = len(tokenizer.vocabulary)
    device = resolve_execution_device(torch_module)
    if options.base_model_path is not None:
        vocab_size = _infer_checkpoint_vocab_size(
            torch_module, options.base_model_path, device, vocab_size,
        )
        _pad_tokenizer_vocabulary(tokenizer, vocab_size)
    model = build_or_load_model(
        torch_module=torch_module,
        base_model=None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, vocab_size),
        device=device,
        initial_weights_path=options.base_model_path,
        training_options=training_options,
    )
    _run_drift_baseline(torch_module, model, options, tokenizer, device)
    precision = build_training_precision_runtime(
        torch_module=torch_module, requested_mode=options.precision_mode, device=device,
    )
    optimization = build_training_optimization(torch_module, model, training_options)
    hooks = load_training_hooks(options.hooks_path)
    run_registry = TrainingRunRegistry(data_root)
    context = TrainingRuntimeContext(
        torch_module=torch_module, model=model,
        optimizer=optimization.optimizer, scheduler=optimization.scheduler,
        precision_runtime=precision,
        loss_function=torch_module.nn.CrossEntropyLoss(ignore_index=0),
        train_batches=train_batches, validation_batches=val_batches,
        tokenizer=tokenizer, options=training_options,
        output_dir=output_dir, device=device, run_id=run_id,
        config_hash=config_hash,
        hooks=hooks, run_registry=run_registry,
    )
    invoke_hook("on_run_start", context.hooks.on_run_start, context)
    loop_result = run_training_loop(context)
    result = _persist_adaptation_outputs(
        context, loop_result, run_id, config_hash, random_seed,
    )
    invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
    return result


def _texts_to_hf_dataset(texts: list[str]) -> Any:
    """Convert texts to a HuggingFace Dataset with 'text' column."""
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def _run_drift_baseline(
    torch_module: Any, model: Any,
    options: DomainAdaptationOptions, tokenizer: Any, device: Any,
) -> None:
    """Compute and log baseline perplexity on reference data."""
    from serve.drift_detection import compute_perplexity

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
    from serve.data_file_reader import read_data_rows

    path = Path(reference_path).expanduser().resolve()
    if not path.exists():
        raise CrucibleServeError(f"Reference data file not found at {path}.")

    texts: list[str] = []
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".parquet"):
        rows = read_data_rows(str(path))
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                prompt = str(row.get("prompt", "")).strip()
                response = str(row.get("response", "")).strip()
                text = f"{prompt}\n{response}" if prompt and response else prompt
            if text:
                texts.append(text)
    else:
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            text = line.strip()
            if text:
                texts.append(text)

    sequences: list[list[int]] = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if len(tokens) >= 2:
            sequences.append(tokens)
    return sequences


def _persist_adaptation_outputs(
    context: Any, loop_result: Any,
    run_id: str, config_hash: str, random_seed: int,
) -> TrainingRunResult:
    """Persist adapted model and training artifacts."""
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
    plot_path = _try_save_plot(
        context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
    )
    repro_path = save_reproducibility_bundle(
        output_dir=context.output_dir, run_id=run_id,
        dataset_name=context.options.dataset_name,
        config_hash=config_hash,
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
    from serve.model_weights import read_model_state_dict

    try:
        state = read_model_state_dict(torch_module, weights_path, device)
        embedding_weight = state.get("embedding.weight")
        if embedding_weight is not None and hasattr(embedding_weight, "shape"):
            return int(embedding_weight.shape[0])
    except CrucibleServeError:
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
    from core.training_types import options_to_training_options
    return options_to_training_options(options, base_model_key="base_model_path")


def _import_torch() -> Any:
    """Import torch dependency for domain adaptation."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "Domain adaptation requires torch, but it is not installed. "
            "Install torch to run crucible domain-adapt."
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
