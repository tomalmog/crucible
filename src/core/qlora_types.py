"""QLoRA training typed models.

This module defines immutable data models for Quantized LoRA
training configuration combining quantization with low-rank adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

DEFAULT_QLORA_BITS = 4
DEFAULT_QLORA_TYPE = "nf4"
DEFAULT_QLORA_DOUBLE_QUANTIZE = True


@dataclass(frozen=True)
class QloraOptions:
    """QLoRA training configuration used by CLI, SDK, and run-spec workflows.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        qlora_data_path: Path to JSONL file with training data.
        base_model_path: Path to the base model to quantize and adapt.
    """

    dataset_name: str
    output_dir: str
    qlora_data_path: str = ""
    base_model_path: str = ""
    quantization_bits: int = DEFAULT_QLORA_BITS
    qlora_type: str = DEFAULT_QLORA_TYPE
    double_quantize: bool = DEFAULT_QLORA_DOUBLE_QUANTIZE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: float = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = 2e-4
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
    hooks_path: str | None = None
    initial_weights_path: str | None = None
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS
