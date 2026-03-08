"""KTO training typed models.

This module defines immutable data models for Kahneman-Tversky
Optimization training using unpaired preference data.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import (
    DEFAULT_BATCH_SIZE,
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

DEFAULT_KTO_BETA = 0.1
DEFAULT_KTO_DESIRABLE_WEIGHT = 1.0
DEFAULT_KTO_UNDESIRABLE_WEIGHT = 1.0


@dataclass(frozen=True)
class KtoExample:
    """One KTO example with binary preference label.

    Attributes:
        prompt: User prompt text.
        response: Model response text.
        is_desirable: True if this is a desirable response.
    """

    prompt: str
    response: str
    is_desirable: bool


@dataclass(frozen=True)
class KtoOptions:
    """KTO training configuration used by CLI, SDK, and run-spec workflows.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        kto_data_path: Path to JSONL with prompt/response/label triples.
    """

    dataset_name: str
    output_dir: str
    kto_data_path: str = ""
    beta: float = DEFAULT_KTO_BETA
    desirable_weight: float = DEFAULT_KTO_DESIRABLE_WEIGHT
    undesirable_weight: float = DEFAULT_KTO_UNDESIRABLE_WEIGHT
    reference_model_path: str | None = None
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = 5e-5
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
