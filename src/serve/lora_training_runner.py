"""LoRA fine-tuning training runner.

This module orchestrates LoRA training: loads base model, injects LoRA
adapters, freezes base parameters, trains only adapter weights,
and persists the adapter alongside training artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeLoraError, ForgeServeError, ForgeTrainingDivergedError
from serve.hf_model_loader import is_huggingface_model_id, load_huggingface_model
from serve.model_format import detect_model_format
from core.lora_types import LoraConfig, LoraTrainingOptions
from core.types import TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.lora_adapter_io import save_lora_adapter
from serve.lora_injection import (
    collect_lora_parameters,
    freeze_base_parameters,
    inject_lora_adapters,
)
from serve.model_weights import load_initial_weights, read_model_state_dict
from serve.sft_data_loader import load_sft_examples
from serve.sft_tokenization import SftSequence, build_sft_sequences
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_model_weights,
    save_training_history,
)
from serve.training_config_hash import compute_training_config_hash
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
from serve.training_progress import emit_progress
from serve.training_setup import validate_file_paths, validate_training_options


def run_lora_training(
    options: LoraTrainingOptions,
    random_seed: int,
    data_root: Path,
    dataset_version_id: str,
) -> TrainingRunResult:
    """Run LoRA fine-tuning and persist adapter artifacts.

    Loads the base model, injects LoRA adapters, freezes base weights,
    trains only adapter parameters, and saves both adapter and base.

    Returns:
        Training run result with artifact paths.

    Raises:
        ForgeLoraError: If LoRA injection or adapter save fails.
        ForgeServeError: If training execution fails.
    """
    _validate_base_model_format(options.base_model_path)
    validate_file_paths(
        base_model_path=options.base_model_path,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        lora_data_path=options.lora_data_path,
    )
    torch_module = _import_torch()
    output_dir = ensure_training_output_dir(options.output_dir)
    run_registry = TrainingRunRegistry(data_root)
    config_hash = compute_training_config_hash_from_lora(options)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        dataset_version_id=dataset_version_id,
        output_dir=str(output_dir),
        parent_model_path=options.base_model_path,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        device = resolve_execution_device(torch_module)
        model = _build_and_load_model(torch_module, options, device)
        lora_config = _resolve_lora_config(torch_module, model, options.lora_config)
        inject_lora_adapters(torch_module, model, lora_config)
        freeze_base_parameters(model)
        lora_params = collect_lora_parameters(model)
        optimizer = _build_lora_optimizer(torch_module, lora_params, options)
        precision_runtime = build_training_precision_runtime(
            torch_module=torch_module,
            requested_mode=options.precision_mode,
            device=device,
        )
        epoch_metrics = _run_lora_training_loop(
            torch_module=torch_module,
            model=model,
            optimizer=optimizer,
            precision_runtime=precision_runtime,
            device=device,
            options=options,
        )
        model_path = save_model_weights(output_dir, torch_module, model)
        adapter_info = save_lora_adapter(
            torch_module, model, output_dir, lora_config,
        )
        history_path = save_training_history(output_dir, epoch_metrics, [])
        result = TrainingRunResult(
            model_path=str(model_path),
            history_path=str(history_path),
            plot_path=None,
            epochs_completed=len(epoch_metrics),
            run_id=run_record.run_id,
        )
        run_registry.transition(
            run_record.run_id, "completed", model_path=str(model_path),
        )
        return result
    except Exception as error:
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


def compute_training_config_hash_from_lora(options: LoraTrainingOptions) -> str:
    """Compute config hash from LoRA training options."""
    import hashlib
    import json

    payload = json.dumps(asdict(options), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "LoRA training requires torch. Install with pip install torch."
        ) from error
    return torch


def _validate_base_model_format(base_model_path: str) -> None:
    """Reject ONNX models early — LoRA requires PyTorch weights."""
    fmt = detect_model_format(base_model_path)
    if fmt == "onnx":
        raise ForgeLoraError(
            f"Cannot LoRA fine-tune an ONNX model ({base_model_path}). "
            "LoRA requires PyTorch weights (.pt). ONNX is an inference-only format — "
            "use a HuggingFace model ID (e.g. 'gpt2') or a .pt checkpoint instead."
        )


def _resolve_lora_config(
    torch_module: Any,
    model: Any,
    config: LoraConfig,
) -> LoraConfig:
    """Auto-detect target modules if the defaults don't match the model.

    Scans the model for linear-like layers (nn.Linear and Conv1D). If none
    of the configured target_modules match any layer names, falls back to
    auto-detected attention/projection layers.
    """
    # Check if configured targets match any layers
    has_match = False
    for name, module in model.named_modules():
        is_linear = isinstance(module, torch_module.nn.Linear)
        is_conv1d = type(module).__name__ == "Conv1D" and hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            continue
        if any(target in name for target in config.target_modules):
            has_match = True
            break

    if has_match:
        return config

    # Auto-detect: find all linear layer name suffixes
    linear_names: set[str] = set()
    for name, module in model.named_modules():
        is_linear = isinstance(module, torch_module.nn.Linear)
        is_conv1d = type(module).__name__ == "Conv1D" and hasattr(module, "nf")
        if is_linear or is_conv1d:
            short_name = name.split(".")[-1]
            linear_names.add(short_name)

    if not linear_names:
        return config

    # Prefer attention layers, exclude output heads
    attention_names = {n for n in linear_names if n != "lm_head"}
    targets = tuple(sorted(attention_names)) if attention_names else tuple(sorted(linear_names))

    print(f"Auto-detected LoRA target modules: {', '.join(targets)}")
    return LoraConfig(
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=targets,
    )


def _infer_checkpoint_vocab_size(
    torch_module: Any, weights_path: str, device: Any, fallback: int,
) -> int:
    """Infer vocab size from checkpoint embedding weights."""
    try:
        state = read_model_state_dict(torch_module, weights_path, device)
        embedding_weight = state.get("embedding.weight")
        if embedding_weight is not None and hasattr(embedding_weight, "shape"):
            return int(embedding_weight.shape[0])
    except Exception:
        pass
    return fallback


def _build_and_load_model(
    torch_module: Any,
    options: LoraTrainingOptions,
    device: Any,
) -> Any:
    """Build or load the base model for LoRA injection.

    Supports three modes:
    - HuggingFace model ID (e.g. 'gpt2'): loads via transformers
    - PyTorch checkpoint (.pt): builds Forge architecture + loads weights
    - Local directory with model files: loads via transformers
    """
    if is_huggingface_model_id(options.base_model_path):
        return load_huggingface_model(options.base_model_path, device=device)

    from core.types import TrainingOptions

    training_options = TrainingOptions(
        dataset_name=options.dataset_name,
        output_dir=options.output_dir,
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        attention_heads=options.attention_heads,
        mlp_hidden_dim=options.mlp_hidden_dim,
        mlp_layers=options.mlp_layers,
        max_token_length=options.max_token_length,
    )
    vocab_size = _infer_checkpoint_vocab_size(
        torch_module, options.base_model_path, device, fallback=10000,
    )
    model = load_training_model(torch_module, training_options, vocab_size=vocab_size)
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.base_model_path,
        device=device,
    )
    return model


def _build_lora_optimizer(
    torch_module: Any,
    lora_params: list[Any],
    options: LoraTrainingOptions,
) -> Any:
    """Build optimizer for LoRA parameters only."""
    if options.optimizer_type == "adamw":
        return torch_module.optim.AdamW(
            lora_params, lr=options.learning_rate, weight_decay=options.weight_decay,
        )
    if options.optimizer_type == "sgd":
        return torch_module.optim.SGD(
            lora_params, lr=options.learning_rate, weight_decay=options.weight_decay,
        )
    return torch_module.optim.Adam(
        lora_params, lr=options.learning_rate, weight_decay=options.weight_decay,
    )


def _load_lora_tokenizer(options: LoraTrainingOptions) -> Any:
    """Load tokenizer for LoRA training.

    Tries in order:
    1. Explicit tokenizer_path from options
    2. HuggingFace AutoTokenizer from the base model ID
    3. Forge vocabulary tokenizer from model directory
    """
    if options.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(options.tokenizer_path)
        except Exception:
            from serve.training_metadata import load_tokenizer_from_path
            return load_tokenizer_from_path(options.tokenizer_path)

    if is_huggingface_model_id(options.base_model_path):
        try:
            from serve.hf_model_loader import load_huggingface_tokenizer
            return load_huggingface_tokenizer(options.base_model_path)
        except Exception:
            pass

    from serve.training_metadata import load_tokenizer
    return load_tokenizer(options.base_model_path)


def _load_lora_sequences(
    data_path: str,
    tokenizer: Any,
    max_token_length: int,
) -> list[SftSequence]:
    """Load LoRA training data with prompt masking via SFT pipeline.

    Expects JSONL with {"prompt": "...", "response": "..."} per line.
    Prompt tokens are masked so the model only learns to predict responses.
    """
    examples = load_sft_examples(data_path)
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=max_token_length,
        mask_prompt_tokens=True,
    )
    if not sequences:
        raise ForgeLoraError(
            "No trainable sequences from LoRA data. "
            "Check data content and max token length."
        )
    return sequences


def _build_lora_batches_from_sequences(
    sequences: list[SftSequence],
    batch_size: int,
) -> list[tuple[list[list[int]], list[list[int]]]]:
    """Group SFT sequences into (inputs, labels) batch pairs."""
    batches: list[tuple[list[list[int]], list[list[int]]]] = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        inputs = [list(s.input_ids) for s in chunk]
        labels = [list(s.labels) for s in chunk]
        batches.append((inputs, labels))
    return batches


def _run_lora_training_loop(
    torch_module: Any,
    model: Any,
    optimizer: Any,
    precision_runtime: Any,
    device: Any,
    options: LoraTrainingOptions,
) -> list[Any]:
    """Run training loop for LoRA fine-tuning.

    Performs real forward/backward passes on tokenized training data.

    Returns:
        List of EpochMetric objects.
    """
    from core.types import EpochMetric

    tokenizer = _load_lora_tokenizer(options)
    if tokenizer is None:
        raise ForgeServeError(
            "No tokenizer found. Provide --tokenizer-path or use a HuggingFace model ID."
        )

    sequences = _load_lora_sequences(
        options.lora_data_path, tokenizer, options.max_token_length,
    )
    batches = _build_lora_batches_from_sequences(sequences, options.batch_size)
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=-100)

    # Enable gradient checkpointing to reduce memory usage.
    if hasattr(model, "gradient_checkpointing_enable"):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    is_mps = str(device).startswith("mps")

    start_epoch = 1
    global_step = 0
    checkpoint_dir = None
    if options.resume_checkpoint_path:
        from serve.training_checkpoint import load_resume_checkpoint
        resume = load_resume_checkpoint(
            options.resume_checkpoint_path, torch_module, model, optimizer, None, device,
        )
        start_epoch = resume.next_epoch
        global_step = resume.global_step

    emit_progress(
        "training_started",
        total_epochs=options.epochs,
        start_epoch=start_epoch,
        method="lora",
    )
    epoch_metrics = []
    model.train()
    for epoch in range(start_epoch, options.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        total_batches = len(batches)
        emit_progress(
            "training_epoch_started",
            epoch=epoch,
            total_epochs=options.epochs,
        )
        for batch_inputs, batch_labels in batches:
            max_len = max(len(s) for s in batch_inputs)
            input_t = torch_module.tensor(
                [s + [0] * (max_len - len(s)) for s in batch_inputs],
                dtype=torch_module.long, device=device,
            )
            label_t = torch_module.tensor(
                [s + [-100] * (max_len - len(s)) for s in batch_labels],
                dtype=torch_module.long, device=device,
            )
            optimizer.zero_grad()
            try:
                logits = model(input_t)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = label_t[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0,
                )
                optimizer.step()
            except RuntimeError as runtime_err:
                if "out of memory" in str(runtime_err).lower():
                    raise ForgeServeError(
                        "GPU out of memory during LoRA training. Try: "
                        "--batch-size (smaller), --max-token-length (shorter), "
                        "or reduce --lora-rank."
                    ) from runtime_err
                raise
            batch_loss = loss.item()
            if math.isnan(batch_loss) or math.isinf(batch_loss):
                raise ForgeTrainingDivergedError(
                    f"LoRA training diverged: loss is {batch_loss} at epoch {epoch}. "
                    "Try reducing --learning-rate or checking your data."
                )
            total_loss += batch_loss
            num_batches += 1
            global_step += 1
            emit_progress(
                "training_batch_progress",
                epoch=epoch,
                total_epochs=options.epochs,
                batch=num_batches,
                total_batches=total_batches,
                loss=round(batch_loss, 6),
            )
            if is_mps:
                torch_module.mps.empty_cache()
        avg_loss = total_loss / max(num_batches, 1)
        epoch_metrics.append(
            EpochMetric(epoch=epoch, train_loss=avg_loss, validation_loss=avg_loss)
        )
        emit_progress(
            "training_epoch_completed",
            epoch=epoch,
            total_epochs=options.epochs,
            train_loss=round(avg_loss, 6),
        )
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = ensure_checkpoint_dir(Path(options.output_dir))
        save_epoch_checkpoint(
            checkpoint_dir, torch_module, model, optimizer, None, epoch, global_step, None,
        )
    return epoch_metrics


