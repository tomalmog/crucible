"""QLoRA training runner for quantized low-rank adaptation.

This module orchestrates QLoRA training: loads a HuggingFace (or local)
base model, applies quantization and LoRA adapters, trains on data,
and persists artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeQloraError, ForgeServeError
from core.lora_types import LoraConfig
from core.qlora_types import QloraOptions
from core.types import DataRecord, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import is_huggingface_model_id, load_huggingface_model
from serve.lora_injection import (
    collect_lora_parameters,
    freeze_base_parameters,
    inject_lora_adapters,
)
from serve.model_weights import load_initial_weights
from serve.quantization_utils import QuantizationConfig, validate_quantization_config
from serve.sft_data_loader import load_sft_examples
from serve.sft_tokenization import build_sft_sequences
from serve.training_artifacts import ensure_training_output_dir
from serve.training_config_hash import compute_training_config_hash
from serve.training_context import TrainingRuntimeContext
from serve.training_execution import run_training_loop
from serve.training_hooks import invoke_hook, load_training_hooks
from serve.training_optimization import build_training_optimization
from serve.training_precision import build_training_precision_runtime
from serve.training_run_registry import TrainingRunRegistry
from serve.training_setup import fit_training_tokenizer, validate_file_paths, validate_training_options


def run_qlora_training(
    records: list[DataRecord],
    options: QloraOptions,
    random_seed: int,
    data_root: Path,
) -> TrainingRunResult:
    """Run a full QLoRA training workflow and persist run lifecycle metadata."""
    quant_config = QuantizationConfig(
        bits=options.quantization_bits,
        quant_type=options.qlora_type,
        double_quantize=options.double_quantize,
    )
    validate_quantization_config(quant_config)
    training_options = _qlora_options_to_training_options(options)
    config_hash = compute_training_config_hash(training_options)
    run_registry = TrainingRunRegistry(data_root)
    run_record = run_registry.start_run(
        dataset_name=options.dataset_name,
        output_dir=str(Path(options.output_dir).expanduser().resolve()),
        parent_model_path=options.base_model_path,
        config_hash=config_hash,
    )
    context: TrainingRuntimeContext | None = None
    try:
        context = _build_qlora_runtime_context(
            records=records,
            options=options,
            training_options=training_options,
            random_seed=random_seed,
            run_id=run_record.run_id,
            config_hash=config_hash,
            run_registry=run_registry,
        )
        run_registry.transition(run_record.run_id, "running")
        invoke_hook("on_run_start", context.hooks.on_run_start, context)
        loop_result = run_training_loop(context)
        result = _persist_qlora_outputs(
            context=context,
            loop_result=loop_result,
            run_id=run_record.run_id,
            config_hash=config_hash,
            random_seed=random_seed,
        )
        invoke_hook("on_run_end", context.hooks.on_run_end, context, result)
        run_registry.transition(
            run_id=run_record.run_id,
            next_state="completed",
            artifact_contract_path=result.artifact_contract_path,
            model_path=result.model_path,
        )
        return result
    except Exception as error:
        if context is not None:
            try:
                invoke_hook(
                    "on_run_error",
                    context.hooks.on_run_error,
                    context,
                    str(error),
                )
            except ForgeServeError:
                pass
        run_registry.transition(
            run_record.run_id, "failed", message=str(error),
        )
        raise


def _build_qlora_runtime_context(
    records: list[DataRecord],
    options: QloraOptions,
    training_options: TrainingOptions,
    random_seed: int,
    run_id: str,
    config_hash: str,
    run_registry: TrainingRunRegistry,
) -> TrainingRuntimeContext:
    """Build runtime context for QLoRA training."""
    torch_module = _import_torch()
    validate_training_options(training_options)
    validate_file_paths(
        base_model_path=options.base_model_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
        tokenizer_path=options.tokenizer_path,
        hooks_path=options.hooks_path,
    )
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, training_options, base_model=options.base_model_path)
    train_batches, val_batches = _build_qlora_batches(
        data_path=options.qlora_data_path,
        tokenizer=tokenizer,
        options=options,
    )
    device = resolve_execution_device(torch_module)
    model = _load_qlora_base_model(
        torch_module=torch_module,
        options=options,
        training_options=training_options,
        tokenizer=tokenizer,
        device=device,
    )
    lora_config = _resolve_qlora_lora_config(torch_module, model, options)
    inject_lora_adapters(torch_module, model, lora_config)
    freeze_base_parameters(model)
    precision_runtime = build_training_precision_runtime(
        torch_module=torch_module,
        requested_mode=options.precision_mode,
        device=device,
    )
    lora_params = collect_lora_parameters(model)
    optimizer = _build_qlora_optimizer(
        torch_module, lora_params, options,
    )
    hooks = load_training_hooks(options.hooks_path)
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=-100)
    return TrainingRuntimeContext(
        torch_module=torch_module,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        precision_runtime=precision_runtime,
        loss_function=loss_fn,
        train_batches=train_batches,
        validation_batches=val_batches,
        tokenizer=tokenizer,
        options=training_options,
        output_dir=output_dir,
        device=device,
        run_id=run_id,
        config_hash=config_hash,
        hooks=hooks,
        run_registry=run_registry,
    )


def _load_qlora_base_model(
    torch_module: Any,
    options: QloraOptions,
    training_options: TrainingOptions,
    tokenizer: Any,
    device: Any,
) -> Any:
    """Load and prepare the base model for QLoRA training.

    Supports HuggingFace model IDs (e.g. 'gpt2') and local checkpoints.
    HuggingFace models are wrapped so forward() returns raw logits,
    compatible with the standard Forge training loop.
    """
    if is_huggingface_model_id(options.base_model_path):
        hf_model = load_huggingface_model(
            options.base_model_path, device=device,
        )
        if hasattr(hf_model, "gradient_checkpointing_enable"):
            if hasattr(hf_model, "enable_input_require_grads"):
                hf_model.enable_input_require_grads()
            hf_model.gradient_checkpointing_enable()
        return _HfLogitsWrapper(torch_module, hf_model)
    model = load_training_model(
        torch_module, training_options, len(tokenizer.vocabulary),
    )
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.base_model_path,
        device=device,
    )
    return model


class _HfLogitsWrapper:
    """Wraps a HuggingFace causal LM so forward returns raw logits.

    The Forge training loop calls model(inputs) and expects a plain
    tensor of logits. HuggingFace models return dataclass-like objects
    with a .logits attribute. This thin wrapper bridges the two.
    """

    def __init__(self, torch_module: Any, hf_model: Any) -> None:
        self._torch_module = torch_module
        self._hf_model = hf_model

    def __call__(self, input_ids: Any) -> Any:
        outputs = self._hf_model(input_ids=input_ids)
        return outputs.logits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hf_model, name)

    def train(self, mode: bool = True) -> Any:
        self._hf_model.train(mode)
        return self

    def eval(self) -> Any:
        self._hf_model.eval()
        return self

    def to(self, device: Any) -> Any:
        self._hf_model.to(device)
        return self


def _resolve_qlora_lora_config(
    torch_module: Any,
    model: Any,
    options: QloraOptions,
) -> LoraConfig:
    """Build LoRA config and auto-detect target modules if needed."""
    config = LoraConfig(
        rank=options.lora_rank,
        alpha=options.lora_alpha,
        dropout=options.lora_dropout,
        target_modules=options.lora_target_modules,
    )
    # Check if configured targets match any layers in the model
    actual_model = model._hf_model if isinstance(model, _HfLogitsWrapper) else model
    has_match = False
    for name, module in actual_model.named_modules():
        is_linear = isinstance(module, torch_module.nn.Linear)
        is_conv1d = (
            type(module).__name__ == "Conv1D" and hasattr(module, "nf")
        )
        if not (is_linear or is_conv1d):
            continue
        if any(target in name for target in config.target_modules):
            has_match = True
            break
    if has_match:
        return config
    # Auto-detect linear layer names
    linear_names: set[str] = set()
    for name, module in actual_model.named_modules():
        is_linear = isinstance(module, torch_module.nn.Linear)
        is_conv1d = (
            type(module).__name__ == "Conv1D" and hasattr(module, "nf")
        )
        if is_linear or is_conv1d:
            short_name = name.split(".")[-1]
            linear_names.add(short_name)
    if not linear_names:
        return config
    attention_names = {n for n in linear_names if n != "lm_head"}
    targets = (
        tuple(sorted(attention_names))
        if attention_names
        else tuple(sorted(linear_names))
    )
    print(f"Auto-detected QLoRA target modules: {', '.join(targets)}")
    return LoraConfig(
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=targets,
    )


def _build_qlora_optimizer(
    torch_module: Any,
    lora_params: list[Any],
    options: QloraOptions,
) -> Any:
    """Build optimizer for QLoRA trainable parameters only."""
    if options.optimizer_type == "adamw":
        return torch_module.optim.AdamW(
            lora_params,
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
    if options.optimizer_type == "sgd":
        return torch_module.optim.SGD(
            lora_params,
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
    return torch_module.optim.Adam(
        lora_params,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )


def _build_qlora_batches(
    data_path: str,
    tokenizer: Any,
    options: QloraOptions,
) -> tuple[list[Any], list[Any]]:
    """Load data with prompt masking and build train/val batches."""
    from serve.tokenization import SequenceBatch

    examples = load_sft_examples(data_path)
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=options.max_token_length,
        mask_prompt_tokens=True,
    )
    if not sequences:
        raise ForgeQloraError(
            "No trainable sequences from QLoRA data. "
            "Check data content and max token length."
        )
    split_idx = max(1, int(len(sequences) * (1.0 - options.validation_split)))
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:] if split_idx < len(sequences) else []

    def to_batches(seqs: list[Any]) -> list[Any]:
        batches = []
        for i in range(0, len(seqs), options.batch_size):
            chunk = seqs[i : i + options.batch_size]
            inputs = [list(s.input_ids) for s in chunk]
            labels = [list(s.labels) for s in chunk]
            batches.append(SequenceBatch(inputs=inputs, targets=labels))
        return batches

    return to_batches(train_seqs), to_batches(val_seqs)


def _persist_qlora_outputs(
    context: TrainingRuntimeContext,
    loop_result: Any,
    run_id: str,
    config_hash: str,
    random_seed: int,
) -> TrainingRunResult:
    """Persist QLoRA training outputs."""
    from dataclasses import asdict

    from serve.training_artifact_contract import save_training_artifact_contract
    from serve.training_artifacts import save_training_history, save_training_plot
    from serve.training_metadata import save_tokenizer_vocabulary, save_training_config
    from serve.training_reproducibility_bundle import save_reproducibility_bundle

    # Save adapter separately, then merge into base for a usable model
    from serve.lora_adapter_io import merge_lora_into_base, save_lora_adapter
    from core.lora_types import LoraConfig
    qlora_config = LoraConfig(rank=getattr(context.options, 'lora_rank', 8))
    save_lora_adapter(context.torch_module, context.model, context.output_dir, qlora_config)
    merged_path = str(context.output_dir / "model.pt")
    merge_lora_into_base(context.torch_module, context.model, merged_path)
    model_path = Path(merged_path)
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
        reproducibility_bundle_path="",
    )
    return TrainingRunResult(
        model_path=base_result.model_path, history_path=base_result.history_path,
        plot_path=base_result.plot_path,
        epochs_completed=base_result.epochs_completed,
        checkpoint_dir=base_result.checkpoint_dir,
        best_checkpoint_path=base_result.best_checkpoint_path,
        run_id=run_id, artifact_contract_path=str(contract_path),
    )


def _qlora_options_to_training_options(
    options: QloraOptions,
) -> TrainingOptions:
    """Map QloraOptions to TrainingOptions."""
    return TrainingOptions(
        dataset_name=options.dataset_name,
        output_dir=options.output_dir,
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
        mlp_hidden_dim=options.mlp_hidden_dim,
        mlp_layers=options.mlp_layers,
        hooks_path=options.hooks_path,
        initial_weights_path=options.base_model_path,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        save_best_checkpoint=options.save_best_checkpoint,
        progress_log_interval_steps=options.progress_log_interval_steps,
        tokenizer_path=options.tokenizer_path,
        resume_checkpoint_path=options.resume_checkpoint_path,
    )


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "QLoRA training requires torch. "
            "Install torch to run forge qlora-train."
        ) from error
    return torch
