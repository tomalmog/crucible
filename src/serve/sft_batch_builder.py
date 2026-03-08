"""SFT batch building and output persistence helpers.

Extracted from sft_runner to keep the runner under the 300-line limit.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError
from core.sft_types import SftOptions
from core.types import TrainingRunResult
from serve.sft_tokenization import SftSequence
from serve.tokenization import SequenceBatch
from serve.training_artifact_contract import save_training_artifact_contract
from serve.training_artifacts import (
    save_model_weights,
    save_training_history,
    save_training_plot,
)
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import TrainingLoopResult
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
from serve.training_reproducibility_bundle import save_reproducibility_bundle


def build_sft_batches(
    sequences: list[SftSequence],
    options: SftOptions,
) -> tuple[list[SequenceBatch], list[SequenceBatch]]:
    """Split SFT sequences and build SequenceBatch lists."""
    input_lists = [list(seq.input_ids) for seq in sequences]
    label_lists = [list(seq.labels) for seq in sequences]
    split_index = _compute_split_index(len(sequences), options.validation_split)
    train_inputs = input_lists[:split_index]
    train_labels = label_lists[:split_index]
    val_inputs = input_lists[split_index:]
    val_labels = label_lists[split_index:]
    train_batches = _batch_sft_pairs(train_inputs, train_labels, options.batch_size)
    val_batches = _batch_sft_pairs(val_inputs, val_labels, options.batch_size)
    return train_batches, val_batches


def persist_sft_outputs(
    context: TrainingRuntimeContext,
    loop_result: TrainingLoopResult,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist model/history/plot outputs and return summary metadata."""
    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    config_path = save_training_config(context.output_dir, context.options)
    tokenizer_path = save_tokenizer_vocabulary(context.output_dir, context.tokenizer)
    history_path = save_training_history(
        context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics,
    )
    plot_path = _try_save_plot(context.output_dir, loop_result.epoch_metrics, loop_result.batch_metrics)
    reproducibility_bundle_path = save_reproducibility_bundle(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.options.dataset_name,
        config_hash=config_hash,
        random_seed=random_seed,
        training_options=asdict(context.options),
    )
    base_result = TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(loop_result.epoch_metrics),
        checkpoint_dir=str(loop_result.checkpoint_dir) if loop_result.checkpoint_dir else None,
        best_checkpoint_path=(
            str(loop_result.best_checkpoint_path) if loop_result.best_checkpoint_path else None
        ),
        resumed_from_checkpoint=loop_result.resumed_from_checkpoint,
        run_id=run_id,
        artifact_contract_path=None,
    )
    contract_path = save_training_artifact_contract(
        output_dir=context.output_dir,
        run_id=run_id,
        dataset_name=context.options.dataset_name,
        parent_model_path=context.options.initial_weights_path,
        config_hash=config_hash,
        result=base_result,
        tokenizer_path=str(tokenizer_path),
        training_config_path=str(config_path),
        reproducibility_bundle_path=str(reproducibility_bundle_path),
    )
    return TrainingRunResult(
        model_path=base_result.model_path,
        history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        resumed_from_checkpoint=base_result.resumed_from_checkpoint,
        run_id=run_id,
        artifact_contract_path=str(contract_path),
    )


def _compute_split_index(total: int, validation_split: float) -> int:
    """Compute the index separating train from validation."""
    validation_size = int(total * validation_split)
    if validation_size < 1:
        validation_size = 1
    if validation_size >= total:
        validation_size = max(total - 1, 0)
    return total - validation_size


def _batch_sft_pairs(
    inputs: list[list[int]],
    labels: list[list[int]],
    batch_size: int,
) -> list[SequenceBatch]:
    """Create SequenceBatch objects from parallel input/label lists."""
    batches: list[SequenceBatch] = []
    batch_inputs: list[list[int]] = []
    batch_targets: list[list[int]] = []
    for inp, lbl in zip(inputs, labels):
        batch_inputs.append(inp)
        batch_targets.append(lbl)
        if len(batch_inputs) >= batch_size:
            batches.append(SequenceBatch(inputs=batch_inputs, targets=batch_targets))
            batch_inputs, batch_targets = [], []
    if batch_inputs:
        batches.append(SequenceBatch(inputs=batch_inputs, targets=batch_targets))
    return batches


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
