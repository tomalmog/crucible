"""Domain adaptation typed models.

This module defines immutable data models for domain adaptation
(continued pretraining) configuration with drift detection settings.
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
class DomainAdaptationOptions:
    """Domain adaptation (continued pretraining) configuration.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        base_model_path: Path to pretrained model weights to continue from.
        reference_data_path: Optional path for drift detection sequences.
        drift_check_interval_epochs: Epochs between perplexity drift checks.
        max_perplexity_increase: Ratio threshold for early stopping.
    """

    dataset_name: str
    output_dir: str
    base_model_path: str
    reference_data_path: str | None = None
    drift_check_interval_epochs: int = 1
    max_perplexity_increase: float = 1.5
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
    hooks_path: str | None = None
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS
