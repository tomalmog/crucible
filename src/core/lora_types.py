"""LoRA adapter typed models.

This module defines immutable data models for LoRA (Low-Rank Adaptation)
configuration including rank, scaling, and target module selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)
from core.training_types import OptimizerType, PrecisionMode


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter hyperparameters.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor (effective scale = alpha / rank).
        dropout: Dropout applied to LoRA input.
        target_modules: Module name substrings to inject LoRA into.
    """

    rank: int = DEFAULT_LORA_RANK
    alpha: float = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES


@dataclass(frozen=True)
class LoraAdapterInfo:
    """Metadata for a saved LoRA adapter.

    Attributes:
        adapter_path: Path to the saved adapter weights.
        rank: Rank used during training.
        alpha: Alpha scaling used during training.
        target_modules: Modules the adapter was injected into.
        base_model_path: Path to the base model the adapter was trained on.
    """

    adapter_path: str
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    base_model_path: str | None = None


@dataclass(frozen=True)
class LoraTrainingOptions:
    """LoRA fine-tuning training options.

    Attributes:
        dataset_name: Dataset to train on.
        output_dir: Directory for output artifacts.
        sft_data_path: Path to SFT JSONL data for LoRA fine-tuning.
        lora_config: LoRA adapter hyperparameters.
        base_model_path: Path to pre-trained model weights.
    """

    dataset_name: str
    output_dir: str
    sft_data_path: str
    base_model_path: str
    lora_config: LoraConfig = LoraConfig()
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    version_id: str | None = None
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    validation_split: float = DEFAULT_TRAIN_VALIDATION_SPLIT
    precision_mode: PrecisionMode = DEFAULT_TRAIN_PRECISION_MODE
    optimizer_type: OptimizerType = DEFAULT_TRAIN_OPTIMIZER_TYPE
    weight_decay: float = DEFAULT_TRAIN_WEIGHT_DECAY
    hidden_dim: int = DEFAULT_TRAIN_HIDDEN_DIM
    num_layers: int = DEFAULT_TRAIN_NUM_LAYERS
    attention_heads: int = DEFAULT_TRAIN_ATTENTION_HEADS
    mlp_hidden_dim: int = DEFAULT_TRAIN_MLP_HIDDEN_DIM
    mlp_layers: int = DEFAULT_TRAIN_MLP_LAYERS
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS
    hooks_path: str | None = None
