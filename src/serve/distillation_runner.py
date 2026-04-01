"""Knowledge distillation training runner.

This module orchestrates distillation: loads teacher model in frozen eval
mode, creates or loads student model, runs blended KL+CE training loop,
and persists the trained student model artifacts.

Distillation keeps the same Crucible implementation for BOTH HF and .pt
paths since trl does not have a distillation trainer. The attention_mask
and unwrap_hf_model fixes are applied for HF models.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_TRAIN_MAX_CHECKPOINT_FILES
from core.distillation_types import DistillationOptions
from core.errors import CrucibleDependencyError, CrucibleDistillationError
from core.training_types import options_to_training_options
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import build_or_load_model, is_huggingface_model_id
from serve.distillation_loss import compute_distillation_loss
from serve.model_weights import load_initial_weights, read_model_state_dict
from core.chat_types import ChatTokenizer
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
from serve.training_config_hash import compute_training_config_hash
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
from serve.training_progress import emit_progress
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options


def run_distillation(
    records: list[DataRecord],
    options: DistillationOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run knowledge distillation training workflow.

    Args:
        records: Dataset records used for tokenizer fitting.
        options: Distillation training options.
        random_seed: Seed for reproducibility.
        data_root: Crucible data root path.

    Returns:
        Training run artifact summary.

    Raises:
        CrucibleDistillationError: If distillation training fails.
        CrucibleDependencyError: If torch is not installed.
    """
    torch_module = _import_torch()
    training_options = _to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.teacher_model_path,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        result = _execute_distillation(
            torch_module=torch_module,
            records=records,
            options=options,
            training_options=training_options,
            run_id=run_record.run_id,
        )
        run_registry.transition(
            run_record.run_id, "completed", model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(
            run_record.run_id, "failed", message=str(error),
        )
        raise


def _execute_distillation(
    torch_module: Any,
    records: list[DataRecord],
    options: DistillationOptions,
    training_options: TrainingOptions,
    run_id: str,
) -> TrainingRunResult:
    """Build models and run distillation loop."""
    validate_training_options(training_options)
    validate_file_paths(
        teacher_model_path=options.teacher_model_path,
        student_model_path=options.student_model_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options)
    vocab_size = len(tokenizer.vocabulary)
    device = resolve_execution_device(torch_module)

    use_hf_teacher = options.teacher_model_path and is_huggingface_model_id(options.teacher_model_path)
    if use_hf_teacher:
        teacher = _load_teacher_model(
            torch_module, training_options, vocab_size, options, device,
        )
        inner = getattr(teacher, "model", teacher)
        config = getattr(inner, "config", None)
        teacher_vocab_size = int(getattr(config, "vocab_size", vocab_size)) if config else vocab_size
    else:
        teacher, teacher_vocab_size, teacher_options = _build_crucible_teacher(
            torch_module, options.teacher_model_path, training_options, device,
        )
    use_hf_student = options.student_model_path and is_huggingface_model_id(options.student_model_path)
    if options.student_model_path and not use_hf_student:
        student, _, _ = _build_crucible_model_from_checkpoint(
            torch_module, options.student_model_path, training_options, device,
        )
    else:
        student = _load_student_model(
            torch_module, training_options, teacher_vocab_size, options, device,
        )
    precision_runtime = build_training_precision_runtime(
        torch_module=torch_module,
        requested_mode=options.precision_mode,
        device=device,
    )
    optimizer = _build_optimizer(torch_module, student, options)
    is_hf = use_hf_teacher or use_hf_student
    epoch_metrics = _run_distillation_loop(
        torch_module=torch_module,
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        precision_runtime=precision_runtime,
        device=device,
        options=options,
        records=records,
        tokenizer=tokenizer,
        teacher_vocab_size=teacher_vocab_size,
        is_hf=is_hf,
    )
    return _persist_outputs(
        output_dir=output_dir,
        torch_module=torch_module,
        student=student,
        epoch_metrics=epoch_metrics,
        run_id=run_id,
        tokenizer=tokenizer,
        training_options=training_options,
        is_hf=is_hf,
    )


def _infer_teacher_vocab_size(
    torch_module: Any, teacher_model_path: str, device: Any,
) -> int:
    """Infer vocabulary size from teacher checkpoint embedding weights."""
    state_dict = read_model_state_dict(torch_module, teacher_model_path, device)
    embedding_weight = state_dict.get("embedding.weight")
    if embedding_weight is None or not hasattr(embedding_weight, "shape"):
        raise CrucibleDistillationError(
            f"Cannot infer vocabulary size from teacher checkpoint at "
            f"{teacher_model_path}: missing embedding.weight tensor."
        )
    return int(embedding_weight.shape[0])


def _build_crucible_model_from_checkpoint(
    torch_module: Any,
    model_path: str,
    base_options: TrainingOptions,
    device: Any,
) -> tuple[Any, int, TrainingOptions]:
    """Build a Crucible model by inferring architecture from a checkpoint."""
    from serve.model_weights import build_and_load_from_checkpoint
    return build_and_load_from_checkpoint(
        torch_module, model_path, base_options, device,
    )


def _build_crucible_teacher(
    torch_module: Any,
    teacher_model_path: str,
    base_options: TrainingOptions,
    device: Any,
) -> tuple[Any, int, TrainingOptions]:
    """Build teacher model from Crucible checkpoint with architecture inference."""
    teacher, vocab_size, inferred_options = _build_crucible_model_from_checkpoint(
        torch_module, teacher_model_path, base_options, device,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher, vocab_size, inferred_options


def _load_teacher_model(
    torch_module: Any,
    training_options: TrainingOptions,
    vocab_size: int,
    options: DistillationOptions,
    device: Any,
) -> Any:
    """Load teacher model in frozen eval mode."""
    use_hf_teacher = options.teacher_model_path and is_huggingface_model_id(options.teacher_model_path)
    teacher = build_or_load_model(
        torch_module=torch_module,
        base_model=options.teacher_model_path if use_hf_teacher else None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, vocab_size),
        device=device,
    )
    if not use_hf_teacher:
        load_initial_weights(
            torch_module=torch_module,
            model=teacher,
            initial_weights_path=options.teacher_model_path,
            device=device,
        )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _load_student_model(
    torch_module: Any,
    training_options: TrainingOptions,
    vocab_size: int,
    options: DistillationOptions,
    device: Any,
) -> Any:
    """Create or load student model for training."""
    use_hf_student = options.student_model_path and is_huggingface_model_id(options.student_model_path)
    student = build_or_load_model(
        torch_module=torch_module,
        base_model=options.student_model_path if use_hf_student else None,
        build_crucible_model=lambda: load_training_model(torch_module, training_options, vocab_size),
        device=device,
    )
    if not use_hf_student and options.student_model_path is not None:
        load_initial_weights(
            torch_module=torch_module,
            model=student,
            initial_weights_path=options.student_model_path,
            device=device,
        )
    return student


def _build_optimizer(
    torch_module: Any, model: Any, options: DistillationOptions,
) -> Any:
    """Build optimizer for student model parameters."""
    params = list(model.parameters())
    if options.optimizer_type == "adamw":
        return torch_module.optim.AdamW(
            params, lr=options.learning_rate, weight_decay=options.weight_decay,
        )
    if options.optimizer_type == "sgd":
        return torch_module.optim.SGD(
            params, lr=options.learning_rate, weight_decay=options.weight_decay,
        )
    return torch_module.optim.Adam(
        params, lr=options.learning_rate, weight_decay=options.weight_decay,
    )


def _unwrap_hf_model(model: Any) -> Any:
    """Unwrap HuggingFace model to get the raw forward-compatible module."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        return model
    return model


def _run_distillation_loop(
    torch_module: Any,
    teacher: Any,
    student: Any,
    optimizer: Any,
    precision_runtime: Any,
    device: Any,
    options: DistillationOptions,
    records: list[DataRecord],
    tokenizer: ChatTokenizer,
    teacher_vocab_size: int,
    is_hf: bool = False,
) -> list[Any]:
    """Run epoch-based distillation training loop."""
    from core.types import EpochMetric

    sequences = build_training_sequences(
        records, tokenizer, options.max_token_length,
    )
    train_seqs, val_seqs = split_sequences(sequences, options.validation_split)
    train_batches = build_sequence_batches(train_seqs, options.batch_size)
    val_batches = build_sequence_batches(val_seqs, options.batch_size)

    start_epoch = 1
    global_step = 0
    if options.resume_checkpoint_path:
        from serve.training_checkpoint import load_resume_checkpoint
        resume = load_resume_checkpoint(
            options.resume_checkpoint_path, torch_module, student,
            optimizer, None, device,
        )
        start_epoch = resume.next_epoch
        global_step = resume.global_step
    emit_progress(
        "training_started",
        total_epochs=options.epochs,
        start_epoch=start_epoch,
        method="distillation",
    )
    epoch_metrics: list[EpochMetric] = []
    epoch = start_epoch
    checkpoint_dir = None
    try:
        for epoch in range(start_epoch, options.epochs + 1):
            emit_progress(
                "training_epoch_started",
                epoch=epoch,
                total_epochs=options.epochs,
            )
            train_loss = _run_distillation_pass(
                torch_module=torch_module,
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                precision_runtime=precision_runtime,
                batches=train_batches,
                device=device,
                options=options,
                teacher_vocab_size=teacher_vocab_size,
                training=True,
                is_hf=is_hf,
            )
            val_loss = _run_distillation_pass(
                torch_module=torch_module,
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                precision_runtime=precision_runtime,
                batches=val_batches,
                device=device,
                options=options,
                teacher_vocab_size=teacher_vocab_size,
                training=False,
                is_hf=is_hf,
            )
            emit_progress(
                "training_epoch_completed",
                epoch=epoch,
                total_epochs=options.epochs,
                train_loss=round(train_loss, 6),
                validation_loss=round(val_loss, 6),
            )
            epoch_metrics.append(
                EpochMetric(
                    epoch=epoch,
                    train_loss=round(train_loss, 6),
                    validation_loss=round(val_loss, 6),
                )
            )
            global_step += 1
            from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir, prune_epoch_checkpoints
            checkpoint_dir = ensure_checkpoint_dir(Path(options.output_dir))
            save_epoch_checkpoint(
                checkpoint_dir, torch_module, student, optimizer, None,
                epoch, global_step, None,
            )
            prune_epoch_checkpoints(checkpoint_dir, max_files=DEFAULT_TRAIN_MAX_CHECKPOINT_FILES)
    except KeyboardInterrupt:
        print("\nTraining interrupted -- saving emergency checkpoint...", flush=True)
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        emergency_dir = checkpoint_dir or ensure_checkpoint_dir(Path(options.output_dir))
        emergency_path = save_epoch_checkpoint(
            emergency_dir, torch_module, student, optimizer, None,
            epoch, global_step, None,
        )
        print(f"Emergency checkpoint saved: {emergency_path}", flush=True)
        raise
    return epoch_metrics


def _run_distillation_pass(
    torch_module: Any,
    teacher: Any,
    student: Any,
    optimizer: Any,
    precision_runtime: Any,
    batches: list[Any],
    device: Any,
    options: DistillationOptions,
    teacher_vocab_size: int,
    training: bool,
    is_hf: bool = False,
) -> float:
    """Run one train or validation pass over distillation batches."""
    if not batches:
        return 0.0
    student.train(mode=training)
    total_loss = 0.0
    for batch in batches:
        inputs, targets = _tensorize_distill_batch(
            torch_module, batch, device, teacher_vocab_size,
        )
        autocast_ctx = _build_autocast_context(
            torch_module, precision_runtime, device,
        )
        with autocast_ctx:
            with torch_module.no_grad():
                if is_hf:
                    attention_mask = (inputs != 0).long()
                    teacher_out = teacher(input_ids=inputs, attention_mask=attention_mask)
                    teacher_logits = teacher_out.logits
                else:
                    teacher_logits = teacher(inputs)
            if is_hf:
                attention_mask = (inputs != 0).long()
                student_out = student(input_ids=inputs, attention_mask=attention_mask)
                student_logits = student_out.logits
            else:
                student_logits = student(inputs)
            # Clamp targets to student vocab for hard CE loss
            student_vocab = student_logits.shape[-1]
            if student_vocab < teacher_vocab_size:
                targets = torch_module.clamp(targets, min=0, max=student_vocab - 1)
            loss = compute_distillation_loss(
                torch_module=torch_module,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=targets,
                temperature=options.temperature,
                alpha=options.alpha,
            )
        loss_value = float(loss.item())
        if training:
            optimizer.zero_grad()
            if precision_runtime.scaler is not None:
                precision_runtime.scaler.scale(loss).backward()
                torch_module.nn.utils.clip_grad_norm_(
                    student.parameters(), max_norm=1.0,
                )
                precision_runtime.scaler.step(optimizer)
                precision_runtime.scaler.update()
            else:
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    student.parameters(), max_norm=1.0,
                )
                optimizer.step()
        total_loss += loss_value
    return total_loss / len(batches)


def _tensorize_distill_batch(
    torch_module: Any, batch: Any, device: Any, vocab_size: int,
) -> tuple[Any, Any]:
    """Convert a SequenceBatch into padded input/target tensors."""
    max_len = max(len(s) for s in batch.inputs)
    padded_in = [s + [0] * (max_len - len(s)) for s in batch.inputs]
    padded_tgt = [s + [0] * (max_len - len(s)) for s in batch.targets]
    input_t = torch_module.tensor(padded_in, dtype=torch_module.long).to(device)
    target_t = torch_module.tensor(padded_tgt, dtype=torch_module.long).to(device)
    target_t = torch_module.clamp(target_t, min=0, max=vocab_size - 1)
    return input_t, target_t


def _build_autocast_context(
    torch_module: Any, precision_runtime: Any, device: Any,
) -> Any:
    """Build autocast context for mixed precision forward pass."""
    if not precision_runtime.autocast_enabled:
        return nullcontext()
    autocast_fn = getattr(torch_module, "autocast", None)
    if autocast_fn is None:
        return nullcontext()
    return autocast_fn(
        device_type=getattr(device, "type", str(device)),
        dtype=precision_runtime.autocast_dtype,
    )


def _persist_outputs(
    output_dir: Path,
    torch_module: Any,
    student: Any,
    epoch_metrics: list[Any],
    run_id: str,
    tokenizer: Any = None,
    training_options: Any = None,
    is_hf: bool = False,
) -> TrainingRunResult:
    """Save student model, training history, tokenizer, and config."""
    if is_hf:
        # For HF models, unwrap and save the underlying model
        inner = getattr(student, "base_model", None)
        if inner is not None and hasattr(inner, "merge_and_unload"):
            student = student.merge_and_unload()
    model_path = save_model_weights(output_dir, torch_module, student)
    history_path = save_training_history(output_dir, epoch_metrics, [])
    if tokenizer is not None:
        from serve.training_metadata import save_tokenizer_vocabulary
        save_tokenizer_vocabulary(output_dir, tokenizer)
    if training_options is not None:
        from serve.training_metadata import save_training_config
        save_training_config(output_dir, training_options)
    return TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=None,
        epochs_completed=len(epoch_metrics),
        run_id=run_id,
    )


def _to_training_options(options: DistillationOptions) -> TrainingOptions:
    """Map DistillationOptions to TrainingOptions for shared components."""
    return options_to_training_options(options, base_model_key="teacher_model_path")


def _import_torch() -> Any:
    """Import torch dependency used by distillation training."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "Distillation training requires torch, but it is not installed. "
            "Install torch to run crucible distill."
        ) from error
    return torch
