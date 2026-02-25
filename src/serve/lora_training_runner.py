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
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_model_weights,
    save_training_history,
)
from serve.training_config_hash import compute_training_config_hash
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
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


def _load_lora_data(data_path: str) -> list[dict[str, str]]:
    """Load training examples from a JSONL file."""
    path = Path(data_path)
    if not path.exists():
        raise ForgeLoraError(f"LoRA data file not found: {data_path}")
    examples: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _build_lora_batches(
    data: list[dict[str, str]],
    tokenizer: Any,
    options: LoraTrainingOptions,
    torch_module: Any,
    device: Any,
) -> list[Any]:
    """Build training batches from LoRA JSONL data."""
    if hasattr(tokenizer, "pad_token_id"):
        batches = []
        for i in range(0, len(data), options.batch_size):
            chunk = data[i : i + options.batch_size]
            texts = [ex.get("text", ex.get("prompt", "") + " " + ex.get("response", "")) for ex in chunk]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=options.max_token_length,
                return_tensors="pt",
            )
            batches.append(encoded["input_ids"].to(device))
        return batches

    all_ids = []
    for ex in data:
        text = ex.get("text", ex.get("prompt", "") + " " + ex.get("response", ""))
        ids = tokenizer.encode(text, options.max_token_length)
        all_ids.append(torch_module.tensor(ids, dtype=torch_module.long, device=device))
    batches = []
    for i in range(0, len(all_ids), options.batch_size):
        chunk = all_ids[i : i + options.batch_size]
        max_len = max(len(ids) for ids in chunk)
        padded = torch_module.stack([
            torch_module.nn.functional.pad(ids, (0, max_len - len(ids)))
            for ids in chunk
        ])
        batches.append(padded)
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

    train_data = _load_lora_data(options.lora_data_path)
    if not train_data:
        raise ForgeLoraError("No training data loaded. Check the lora_data_path file.")

    tokenizer = _load_lora_tokenizer(options)
    if tokenizer is None:
        raise ForgeServeError(
            "No tokenizer found. Provide --tokenizer-path or use a HuggingFace model ID."
        )

    batches = _build_lora_batches(train_data, tokenizer, options, torch_module, device)

    # Enable gradient checkpointing to reduce memory usage.
    # For LoRA, the embedding inputs don't require grad by default, which
    # breaks gradient checkpointing. enable_input_require_grads() fixes this
    # by ensuring that the inputs to checkpointed segments carry gradients.
    if hasattr(model, "gradient_checkpointing_enable"):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    uses_labels = hasattr(model, "forward") and "labels" in _get_forward_params(model)
    is_mps = str(device).startswith("mps")

    start_epoch = 1
    global_step = 0
    checkpoint_dir = None
    if options.resume_checkpoint_path:
        from serve.training_checkpoint import load_resume_checkpoint, ensure_checkpoint_dir
        resume = load_resume_checkpoint(
            options.resume_checkpoint_path, torch_module, model, optimizer, None, device,
        )
        start_epoch = resume.next_epoch
        global_step = resume.global_step

    epoch_metrics = []
    model.train()
    for epoch in range(start_epoch, options.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for input_ids in batches:
            optimizer.zero_grad()
            try:
                if uses_labels:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                else:
                    logits = model(input_ids)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss_fn = torch_module.nn.CrossEntropyLoss()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                loss.backward()
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
                    "Try reducing --learning-rate, checking your data for corruption, or using --gradient-clipping."
                )
            total_loss += batch_loss
            num_batches += 1
            global_step += 1
            # Free MPS memory between batches
            if is_mps:
                torch_module.mps.empty_cache()
        avg_loss = total_loss / max(num_batches, 1)
        epoch_metrics.append(
            EpochMetric(epoch=epoch, train_loss=avg_loss, validation_loss=avg_loss)
        )
        print(f"Epoch {epoch}/{options.epochs} - loss: {avg_loss:.4f}")
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = ensure_checkpoint_dir(Path(options.output_dir))
        save_epoch_checkpoint(
            checkpoint_dir, torch_module, model, optimizer, None, epoch, global_step, None,
        )
    return epoch_metrics


def _get_forward_params(model: Any) -> set[str]:
    """Get parameter names accepted by model.forward."""
    import inspect
    try:
        sig = inspect.signature(model.forward)
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return set()
