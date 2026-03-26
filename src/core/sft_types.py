"""SFT training typed models.

This module defines immutable data models for supervised fine-tuning
configuration including prompt/response masking and sequence packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_SFT_MASK_PROMPT_TOKENS,
    DEFAULT_SFT_PACKING_ENABLED,
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


@dataclass(frozen=True)
class SftExample:
    """One prompt/response pair for supervised fine-tuning.

    Attributes:
        prompt: User prompt text.
        response: Expected model response text.
        system_prompt: Optional system instruction prefix.
    """

    prompt: str
    response: str
    system_prompt: str | None = None


@dataclass(frozen=True)
class SftOptions:
    """SFT training command options used by CLI, SDK, and run-spec workflows.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        sft_data_path: Path to JSONL file with prompt/response pairs.
    """

    dataset_name: str
    output_dir: str
    sft_data_path: str = ""
    mask_prompt_tokens: bool = DEFAULT_SFT_MASK_PROMPT_TOKENS
    packing_enabled: bool = DEFAULT_SFT_PACKING_ENABLED
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = 2e-5
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
    base_model: str | None = None
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS
