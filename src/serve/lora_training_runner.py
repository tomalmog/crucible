"""LoRA fine-tuning training runner.

This module orchestrates LoRA training: loads base model, injects LoRA
adapters, freezes base parameters, trains only adapter weights,
and persists the adapter alongside training artifacts.

Uses trl.SFTTrainer + peft for HuggingFace models, and falls back to
the Crucible custom training loop for .pt models.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.chat_types import ChatTokenizer
from core.errors import CrucibleDependencyError, CrucibleLoraError, CrucibleServeError, CrucibleTrainingDivergedError
from core.lora_types import LoraConfig, LoraTrainingOptions
from core.types import TrainingRunResult
from serve.model_format import detect_model_format
from serve.sft_data_loader import load_sft_examples
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


def run_lora_training(
    options: LoraTrainingOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run LoRA fine-tuning and persist adapter artifacts.

    Uses trl.SFTTrainer + peft for HuggingFace models. Falls back to
    the Crucible custom loop for .pt checkpoints.

    Returns:
        Training run result with artifact paths.

    Raises:
        CrucibleLoraError: If LoRA injection or adapter save fails.
        CrucibleServeError: If training execution fails.
    """
    _validate_base_model_format(options.base_model_path)
    output_dir = ensure_training_output_dir(options.output_dir)
    run_registry = TrainingRunRegistry(data_root)
    config_hash = compute_training_config_hash_from_lora(options)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(output_dir),
        parent_model_path=options.base_model_path,
        config_hash=config_hash,
    )
    try:
        run_registry.transition(run_record.run_id, "running")
        if options.base_model_path and is_hf_model(options.base_model_path):
            result = _run_lora_with_trl(options, output_dir, run_record.run_id, random_seed)
        else:
            result = _run_lora_crucible(options, output_dir, run_record.run_id, random_seed)
        run_registry.transition(
            run_record.run_id, "completed", model_path=result.model_path,
        )
        return result
    except Exception as error:
        run_registry.transition(run_record.run_id, "failed", message=str(error))
        raise


# ── trl + peft path (HuggingFace models) ────────────────────────────


def _run_lora_with_trl(
    options: LoraTrainingOptions,
    output_dir: Path,
    run_id: str,
    random_seed: int,
) -> TrainingRunResult:
    """LoRA fine-tuning using trl.SFTTrainer + peft for HuggingFace models."""
    from peft import LoraConfig as PeftLoraConfig
    trl = _import_trl()

    model, tokenizer = load_hf_model_and_tokenizer(
        options.base_model_path, options.precision_mode,
    )

    sft_examples = load_sft_examples(options.lora_data_path)
    if not sft_examples:
        raise CrucibleLoraError("No LoRA training examples found.")
    dataset = _examples_to_hf_dataset(sft_examples)
    split = dataset.train_test_split(test_size=options.validation_split, seed=random_seed)

    # Build peft LoraConfig from Crucible LoraConfig.
    # If the user-specified target_modules don't exist in the model
    # (e.g. default "q_proj"/"v_proj" on a GPT-2 model that uses "c_attn"),
    # fall back to "all-linear" which lets peft auto-detect all Linear layers.
    target_mods: list[str] | str = list(options.lora_config.target_modules)
    model_names = {name for name, _ in model.named_modules()}
    if not any(any(t in n for t in target_mods) for n in model_names):
        target_mods = "all-linear"
    lora_config = PeftLoraConfig(
        r=options.lora_config.rank,
        lora_alpha=options.lora_config.alpha,
        lora_dropout=options.lora_config.dropout,
        target_modules=target_mods,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = build_base_training_args(
        output_dir=output_dir,
        epochs=options.epochs,
        batch_size=options.batch_size,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        precision_mode=options.precision_mode,
        log_steps=options.progress_log_interval_steps,
        seed=random_seed,
        max_length=options.max_token_length,
    )
    sft_config = trl.SFTConfig(**args)

    print("LoRA: Starting trl training...", flush=True)
    trainer = trl.SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    # Save peft adapter config for compatibility
    _save_adapter_config_json(output_dir, options.lora_config)

    print("LoRA: Saving model...", flush=True)
    from core.training_types import options_to_training_options
    training_options = options_to_training_options(options, base_model_key="base_model_path")
    return save_trl_outputs(trainer, output_dir, training_options, tokenizer, run_id, options.epochs)


def _examples_to_hf_dataset(examples: list[Any]) -> Any:
    """Convert SftExample list to a HuggingFace Dataset."""
    from datasets import Dataset
    texts = []
    for ex in examples:
        text = f"{ex.system_prompt}\n\n" if ex.system_prompt else ""
        text += f"{ex.prompt}\n{ex.response}"
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def _save_adapter_config_json(output_dir: Path, lora_config: LoraConfig) -> None:
    """Save LoRA adapter config JSON for compatibility with downstream tooling."""
    config_data = {
        "lora_rank": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "lora_dropout": lora_config.dropout,
        "lora_target_modules": list(lora_config.target_modules),
    }
    config_path = output_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")


# ── Crucible .pt path (existing custom loop) ────────────────────────


def _run_lora_crucible(
    options: LoraTrainingOptions,
    output_dir: Path,
    run_id: str,
    random_seed: int,
) -> TrainingRunResult:
    """LoRA fine-tuning using the Crucible custom training loop for .pt models."""
    from serve.hf_model_loader import is_huggingface_model_id, load_huggingface_model
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.lora_adapter_io import save_lora_adapter
    from serve.lora_injection import (
        collect_lora_parameters,
        freeze_base_parameters,
        inject_lora_adapters,
    )
    from serve.model_weights import read_model_state_dict
    from serve.sft_tokenization import SftSequence, build_sft_sequences
    from serve.training_artifacts import save_model_weights, save_training_history
    from serve.training_precision import build_training_precision_runtime
    from serve.training_progress import emit_progress
    from serve.training_setup import validate_file_paths, validate_training_options

    validate_file_paths(
        base_model_path=options.base_model_path,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        lora_data_path=options.lora_data_path,
    )
    torch_module = _import_torch()
    device = resolve_execution_device(torch_module)
    model, inferred_options = _build_and_load_model(torch_module, options, device)
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
    tokenizer = _load_lora_tokenizer(options)
    if tokenizer is None:
        raise CrucibleServeError(
            "No tokenizer found. Provide --tokenizer-path or use a HuggingFace model ID."
        )
    epoch_metrics = _run_lora_training_loop(
        torch_module=torch_module,
        model=model,
        optimizer=optimizer,
        precision_runtime=precision_runtime,
        device=device,
        options=options,
        tokenizer=tokenizer,
    )
    adapter_info = save_lora_adapter(
        torch_module, model, output_dir, lora_config,
    )
    # Merge LoRA adapters into base weights so the saved model
    # can be loaded and used directly without a separate merge step
    from serve.lora_adapter_io import merge_lora_into_base
    merged_path = str(output_dir / "model.pt")
    merge_lora_into_base(torch_module, model, merged_path)
    model_path = Path(merged_path)
    # Save tokenizer and training config alongside the model
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    if tokenizer is not None:
        save_tokenizer_vocabulary(output_dir, tokenizer)
    save_training_config(output_dir, inferred_options or options)
    history_path = save_training_history(output_dir, epoch_metrics, [])
    return TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=None,
        epochs_completed=len(epoch_metrics),
        run_id=run_id,
    )


# ── Shared helpers (used by Crucible .pt path) ──────────────────────


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
        raise CrucibleDependencyError(
            "LoRA training requires torch. Install with pip install torch."
        ) from error
    return torch


def _validate_base_model_format(base_model_path: str) -> None:
    """Reject ONNX models early -- LoRA requires PyTorch weights."""
    fmt = detect_model_format(base_model_path)
    if fmt == "onnx":
        raise CrucibleLoraError(
            f"Cannot LoRA fine-tune an ONNX model ({base_model_path}). "
            "LoRA requires PyTorch weights (.pt). ONNX is an inference-only format -- "
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


def _build_and_load_model(
    torch_module: Any,
    options: LoraTrainingOptions,
    device: Any,
) -> tuple[Any, Any]:
    """Build or load the base model for LoRA injection.

    Returns:
        Tuple of (model, inferred_training_options).  The training options
        reflect the actual architecture of the loaded checkpoint and should
        be used when saving training_config.json.

    Supports three modes:
    - HuggingFace model ID (e.g. 'gpt2'): loads via transformers
    - PyTorch checkpoint (.pt): builds Crucible architecture + loads weights
    - Local directory with model files: loads via transformers
    """
    from serve.hf_model_loader import is_huggingface_model_id, load_huggingface_model
    if is_huggingface_model_id(options.base_model_path):
        return load_huggingface_model(options.base_model_path, device=device), None

    from pathlib import Path

    from core.types import TrainingOptions
    from serve.model_weights import read_model_state_dict
    from serve.architecture_loader import load_training_model

    # Read checkpoint state dict once -- used both for architecture
    # inference and weight loading to avoid reading the file twice.
    state = read_model_state_dict(torch_module, options.base_model_path, device)

    # Infer architecture params from checkpoint to avoid shape mismatches.
    # Without this, a model saved with mlp_layers>1 (Sequential output head)
    # would fail to load into a model built with mlp_layers=1 (single Linear).
    from serve.chat_option_resolver import (
        _infer_encoder_layer_count,
        _infer_projection_layer_count,
        _infer_shape_value,
    )

    vocab_size = _infer_shape_value(state, "embedding.weight", 0) or 10000
    hidden_dim = _infer_shape_value(state, "embedding.weight", 1) or options.hidden_dim
    num_layers = _infer_encoder_layer_count(state) or options.num_layers
    mlp_hidden_dim = (
        _infer_shape_value(state, "encoder.layers.0.linear1.weight", 0)
        or _infer_shape_value(state, "blocks.0.ffn.0.weight", 0)
        or options.mlp_hidden_dim
    )
    mlp_layers = _infer_projection_layer_count(state) or options.mlp_layers
    attention_heads = options.attention_heads
    if attention_heads > 0 and hidden_dim % attention_heads != 0:
        for candidate in range(min(attention_heads, hidden_dim), 0, -1):
            if hidden_dim % candidate == 0:
                attention_heads = candidate
                break

    training_options = TrainingOptions(
        dataset_name=options.dataset_name,
        output_dir=options.output_dir,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        attention_heads=attention_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        max_token_length=options.max_token_length,
    )
    model = load_training_model(torch_module, training_options, vocab_size=vocab_size)
    model = model.to(device)

    # Apply the already-loaded state dict directly.
    from serve.model_weights import _apply_model_state
    _apply_model_state(model, state, Path(options.base_model_path))

    return model, training_options


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


def _load_lora_tokenizer(options: LoraTrainingOptions) -> ChatTokenizer | None:
    """Load tokenizer for LoRA training via the shared resolve_tokenizer helper.

    Tries in order:
    1. Explicit tokenizer_path from options
    2. HuggingFace AutoTokenizer from the base model ID
    3. Crucible vocabulary tokenizer from model directory
    """
    from serve.training_setup import resolve_tokenizer

    resolved = resolve_tokenizer(options.tokenizer_path, options.base_model_path)
    if resolved is not None:
        return resolved

    from serve.training_metadata import load_tokenizer
    return load_tokenizer(options.base_model_path)


def _load_lora_sequences(
    data_path: str,
    tokenizer: ChatTokenizer,
    max_token_length: int,
) -> list[Any]:
    """Load LoRA training data with prompt masking via SFT pipeline.

    Expects JSONL with {"prompt": "...", "response": "..."} per line.
    Prompt tokens are masked so the model only learns to predict responses.
    """
    from serve.sft_tokenization import build_sft_sequences
    examples = load_sft_examples(data_path)
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=max_token_length,
        mask_prompt_tokens=True,
    )
    if not sequences:
        raise CrucibleLoraError(
            "No trainable sequences from LoRA data. "
            "Check data content and max token length."
        )
    return sequences


def _build_lora_batches_from_sequences(
    sequences: list[Any],
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
    tokenizer: ChatTokenizer | None = None,
) -> list[Any]:
    """Run training loop for LoRA fine-tuning.

    Performs real forward/backward passes on tokenized training data.

    Returns:
        List of EpochMetric objects.
    """
    from core.types import EpochMetric
    from serve.training_progress import emit_progress

    if tokenizer is None:
        tokenizer = _load_lora_tokenizer(options)
    if tokenizer is None:
        raise CrucibleServeError(
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
    epoch = start_epoch
    model.train()
    try:
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
                    if getattr(model, "_is_hf_logits_wrapper", False):
                        attn_mask = (input_t != 0).long()
                        logits = model(input_t, attention_mask=attn_mask)
                    else:
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
                        raise CrucibleServeError(
                            "GPU out of memory during LoRA training. Try: "
                            "--batch-size (smaller), --max-token-length (shorter), "
                            "or reduce --lora-rank."
                        ) from runtime_err
                    raise
                batch_loss = loss.item()
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    raise CrucibleTrainingDivergedError(
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
    except KeyboardInterrupt:
        print("\nTraining interrupted -- saving emergency checkpoint...", flush=True)
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        emergency_dir = checkpoint_dir or ensure_checkpoint_dir(Path(options.output_dir))
        emergency_path = save_epoch_checkpoint(
            emergency_dir, torch_module, model, optimizer, None, epoch, global_step, None,
        )
        print(f"Emergency checkpoint saved: {emergency_path}", flush=True)
        raise
    return epoch_metrics
