"""DDP-wrapped training runner for multi-GPU distributed training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeDistributedError
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.distributed_data_sampler import (
    partition_batches_for_rank,
    select_rank_batches,
)
from serve.distributed_setup import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    select_ddp_backend,
)


def run_ddp_training(
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run DDP-wrapped training for multi-GPU via torchrun."""
    torch_module = _import_torch()
    try:
        backend = select_ddp_backend(torch_module)
        init_distributed(torch_module, backend=backend)
        rank = get_rank(torch_module)
        world_size = get_world_size(torch_module)
        device = _resolve_rank_device(torch_module, rank)
        result = _run_ddp_workflow(
            torch_module=torch_module,
            records=records,
            options=options,
            random_seed=random_seed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        return result
    finally:
        cleanup_distributed(torch_module)


def _run_ddp_workflow(
    torch_module: Any,
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
    rank: int,
    world_size: int,
    device: Any,
) -> TrainingRunResult:
    """Build model, wrap with DDP, partition batches, train, and save on rank 0."""
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_device_for_rank
    from serve.tokenization import (
        build_sequence_batches,
        build_training_sequences,
        split_sequences,
    )
    from serve.training_artifacts import (
        ensure_training_output_dir,
        save_model_weights,
        save_training_history,
    )
    from serve.training_setup import (
        fit_training_tokenizer,
        validate_training_options,
    )

    validate_training_options(options)
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, options)
    sequences = build_training_sequences(
        records, tokenizer, options.max_token_length,
    )
    if not sequences:
        raise ForgeDistributedError(
            "No trainable sequences generated from dataset records."
        )
    random.Random(random_seed).shuffle(sequences)
    train_seqs, val_seqs = split_sequences(
        sequences, options.validation_split,
    )
    train_batches = build_sequence_batches(train_seqs, options.batch_size)
    val_batches = build_sequence_batches(val_seqs, options.batch_size)

    partition = partition_batches_for_rank(
        len(train_batches), rank, world_size,
    )
    rank_train_batches = select_rank_batches(train_batches, partition)

    model = load_training_model(
        torch_module, options, len(tokenizer.vocabulary),
    )
    model = model.to(device)
    model = _wrap_model_with_ddp(torch_module, model, rank)

    epoch_metrics = _run_ddp_epochs(
        torch_module=torch_module,
        model=model,
        train_batches=rank_train_batches,
        val_batches=val_batches,
        options=options,
        device=device,
    )

    return _save_rank_zero_artifacts(
        torch_module=torch_module,
        model=model,
        epoch_metrics=epoch_metrics,
        output_dir=output_dir,
        rank=rank,
    )


def _wrap_model_with_ddp(
    torch_module: Any, model: Any, rank: int,
) -> Any:
    """Wrap a model with DistributedDataParallel."""
    nn_parallel = getattr(torch_module.nn, "parallel", None)
    if nn_parallel is None:
        raise ForgeDistributedError(
            "torch.nn.parallel is unavailable for DDP wrapping."
        )
    ddp_class = getattr(nn_parallel, "DistributedDataParallel", None)
    if ddp_class is None:
        raise ForgeDistributedError(
            "DistributedDataParallel is not available in torch.nn.parallel."
        )
    return ddp_class(model, device_ids=[rank])


def _run_ddp_epochs(
    torch_module: Any,
    model: Any,
    train_batches: list[Any],
    val_batches: list[Any],
    options: TrainingOptions,
    device: Any,
) -> list[dict[str, float]]:
    """Run training epochs and collect metrics."""
    optimizer = torch_module.optim.Adam(
        model.parameters(), lr=options.learning_rate,
    )
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=0)
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, options.epochs + 1):
        train_loss = _run_ddp_epoch_pass(
            torch_module=torch_module,
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            training=True,
        )
        val_loss = _run_ddp_epoch_pass(
            torch_module=torch_module,
            model=model,
            batches=val_batches,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            training=False,
        )
        epoch_metrics.append({
            "epoch": float(epoch),
            "train_loss": round(train_loss, 6),
            "validation_loss": round(val_loss, 6),
        })
    return epoch_metrics


def _run_ddp_epoch_pass(
    torch_module: Any,
    model: Any,
    batches: list[Any],
    optimizer: Any,
    loss_fn: Any,
    device: Any,
    training: bool,
) -> float:
    """Run one epoch pass (train or validation)."""
    if not batches:
        return 0.0
    model.train(mode=training)
    total_loss = 0.0
    for batch in batches:
        inputs, targets = _tensorize_ddp_batch(
            torch_module, batch, device,
        )
        logits = model(inputs)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        total_loss += float(loss.item())
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss / len(batches)


def _tensorize_ddp_batch(
    torch_module: Any, batch: Any, device: Any,
) -> tuple[Any, Any]:
    """Convert batch to padded tensors on the target device."""
    max_len = max(len(s) for s in batch.inputs)
    pad_inputs = [s + [0] * (max_len - len(s)) for s in batch.inputs]
    pad_targets = [s + [0] * (max_len - len(s)) for s in batch.targets]
    inp = torch_module.tensor(pad_inputs, dtype=torch_module.long)
    tgt = torch_module.tensor(pad_targets, dtype=torch_module.long)
    return inp.to(device), tgt.to(device)


def _save_rank_zero_artifacts(
    torch_module: Any,
    model: Any,
    epoch_metrics: list[dict[str, float]],
    output_dir: Path,
    rank: int,
) -> TrainingRunResult:
    """Save training artifacts only on rank 0."""
    from core.types import EpochMetric
    from serve.training_artifacts import (
        save_model_weights,
        save_training_history,
    )

    metrics = [
        EpochMetric(
            epoch=int(m["epoch"]),
            train_loss=m["train_loss"],
            validation_loss=m["validation_loss"],
        )
        for m in epoch_metrics
    ]
    if rank == 0:
        unwrapped = _unwrap_ddp_model(model)
        model_path = save_model_weights(
            output_dir, torch_module, unwrapped,
        )
        history_path = save_training_history(
            output_dir, metrics, [],
        )
        return TrainingRunResult(
            model_path=str(model_path),
            history_path=str(history_path),
            plot_path=None,
            epochs_completed=len(metrics),
        )
    return TrainingRunResult(
        model_path="",
        history_path="",
        plot_path=None,
        epochs_completed=len(metrics),
    )


def _unwrap_ddp_model(model: Any) -> Any:
    """Unwrap a DDP model to access the underlying module."""
    inner = getattr(model, "module", None)
    if inner is not None:
        return inner
    return model


def _resolve_rank_device(torch_module: Any, rank: int) -> Any:
    """Resolve device for a specific DDP rank."""
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device(f"cuda:{rank}")
    return torch_module.device("cpu")


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Distributed training requires torch. "
            "Install torch to run forge distributed-train."
        ) from error
    return torch
