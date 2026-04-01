"""Shared base for trl-based training runners.

Extracts the common boilerplate that every trl runner needs:
run registry lifecycle, config hash, output dir creation,
model/tokenizer loading, artifact saving, and history extraction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError
from core.types import TrainingRunResult
from core.training_types import TrainingOptions
from serve.training_artifacts import ensure_training_output_dir
from serve.training_metadata import save_training_config


def load_hf_model_and_tokenizer(
    model_id: str,
    precision_mode: str,
) -> tuple[Any, Any]:
    """Load a HuggingFace model and tokenizer for training."""
    torch = _import_torch()
    transformers = _import_transformers()

    print(f"Loading model {model_id}...", flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_resolve_dtype(torch, precision_mode),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_base_training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    precision_mode: str,
    log_steps: int,
    seed: int,
    max_length: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build kwargs dict for trl config classes (SFTConfig, DPOConfig, etc.)."""
    torch = _import_torch()
    args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "logging_steps": max(log_steps, 1),
        "seed": seed,
        "bf16": precision_mode in ("bf16", "auto") and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": precision_mode == "fp16",
        "max_grad_norm": 1.0,
        "report_to": "none",
        "dataloader_pin_memory": False,
    }
    if max_length is not None:
        args["max_length"] = max_length
    args.update(extra)
    return args


def save_trl_outputs(
    trainer: Any,
    output_dir: Path,
    training_options: TrainingOptions,
    tokenizer: Any,
    run_id: str,
    epochs: int,
) -> TrainingRunResult:
    """Save model, tokenizer, config, and history from a trl Trainer."""
    torch = _import_torch()

    # Save in HF-native format
    hf_dir = output_dir / "hf_model"
    trainer.save_model(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    # Save model.pt for Crucible compatibility
    model = trainer.model
    # Unwrap PEFT if needed
    inner = getattr(model, "base_model", None)
    if inner is not None and hasattr(inner, "merge_and_unload"):
        model = model.merge_and_unload()
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), str(model_path))

    # Save Crucible metadata
    save_training_config(output_dir, training_options)
    tokenizer.save_pretrained(str(output_dir))

    # Extract and save history
    history = extract_training_history(trainer)
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    print(f"Training complete. Model saved to {model_path}", flush=True)
    return TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=None,
        epochs_completed=epochs,
        run_id=run_id,
    )


def extract_training_history(trainer: Any) -> dict[str, Any]:
    """Extract epoch metrics from any trl/transformers Trainer."""
    epoch_data: dict[int, dict[str, float]] = {}
    for entry in trainer.state.log_history:
        if "epoch" not in entry:
            continue
        raw_epoch = float(entry["epoch"])
        epoch_num = int(math.ceil(raw_epoch))
        if epoch_num < 1:
            continue
        if epoch_num not in epoch_data:
            epoch_data[epoch_num] = {
                "epoch": epoch_num,
                "train_loss": 0.0,
                "validation_loss": 0.0,
            }
        if "loss" in entry:
            epoch_data[epoch_num]["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            epoch_data[epoch_num]["validation_loss"] = entry["eval_loss"]
    # Fallback: use train_loss from final summary
    for entry in trainer.state.log_history:
        if "train_loss" in entry and "epoch" in entry:
            epoch_num = int(math.ceil(float(entry["epoch"])))
            if epoch_num in epoch_data and epoch_data[epoch_num]["train_loss"] == 0.0:
                epoch_data[epoch_num]["train_loss"] = entry["train_loss"]
    return {"epochs": [epoch_data[k] for k in sorted(epoch_data)]}


def is_hf_model(model_id: str) -> bool:
    """Check if model_id is a HuggingFace model."""
    from serve.hf_model_loader import is_huggingface_model_id
    return is_huggingface_model_id(model_id)


def _resolve_dtype(torch: Any, precision_mode: str) -> Any:
    if precision_mode == "bf16":
        return torch.bfloat16
    if precision_mode == "fp16":
        return torch.float16
    if precision_mode == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return torch.float32


def _import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as error:
        raise CrucibleDependencyError("Training requires torch.") from error


def _import_transformers() -> Any:
    try:
        import transformers
        return transformers
    except ImportError as error:
        raise CrucibleDependencyError("HuggingFace training requires transformers.") from error


def _import_trl() -> Any:
    try:
        import trl
        return trl
    except ImportError as error:
        raise CrucibleDependencyError("HuggingFace training requires trl. Install with: pip install trl") from error
