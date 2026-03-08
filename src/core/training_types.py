"""Training-specific typed models.

This module defines immutable data models for training configuration,
metrics, and results used by training runners and CLI commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_GRADIENT_CHECKPOINTING_ENABLED,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_POSITION_EMBEDDING_TYPE,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MAX_CHECKPOINT_FILES,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_SCHEDULER_ETA_MIN,
    DEFAULT_TRAIN_SCHEDULER_GAMMA,
    DEFAULT_TRAIN_SCHEDULER_STEP_SIZE,
    DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS,
    DEFAULT_TRAIN_SCHEDULER_TYPE,
    DEFAULT_TRAIN_SGD_MOMENTUM,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)

PositionEmbeddingType = Literal["learned", "sinusoidal"]
PrecisionMode = Literal["auto", "fp32", "fp16", "bf16"]
OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["none", "step", "cosine"]


@dataclass(frozen=True)
class DataLoaderOptions:
    """PyTorch serving options.

    Attributes:
        batch_size: Number of tokenized records per batch.
        shuffle: Whether to shuffle records before yielding.
        shuffle_buffer_size: Buffer size for shuffle algorithm.
        max_token_length: Truncation length for tokenized records.
    """

    batch_size: int
    shuffle: bool
    shuffle_buffer_size: int
    max_token_length: int


@dataclass(frozen=True)
class TrainingOptions:
    """Training command options used by CLI, SDK, and run-spec workflows."""

    dataset_name: str
    output_dir: str
    architecture_path: str | None = None
    custom_loop_path: str | None = None
    hooks_path: str | None = None
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE
    precision_mode: PrecisionMode = DEFAULT_TRAIN_PRECISION_MODE
    optimizer_type: OptimizerType = DEFAULT_TRAIN_OPTIMIZER_TYPE
    weight_decay: float = DEFAULT_TRAIN_WEIGHT_DECAY
    sgd_momentum: float = DEFAULT_TRAIN_SGD_MOMENTUM
    scheduler_type: SchedulerType = DEFAULT_TRAIN_SCHEDULER_TYPE
    scheduler_step_size: int = DEFAULT_TRAIN_SCHEDULER_STEP_SIZE
    scheduler_gamma: float = DEFAULT_TRAIN_SCHEDULER_GAMMA
    scheduler_t_max_epochs: int | None = DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS
    scheduler_eta_min: float = DEFAULT_TRAIN_SCHEDULER_ETA_MIN
    batch_size: int = DEFAULT_BATCH_SIZE
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    validation_split: float = DEFAULT_TRAIN_VALIDATION_SPLIT
    hidden_dim: int = DEFAULT_TRAIN_HIDDEN_DIM
    num_layers: int = DEFAULT_TRAIN_NUM_LAYERS
    attention_heads: int = DEFAULT_TRAIN_ATTENTION_HEADS
    mlp_hidden_dim: int = DEFAULT_TRAIN_MLP_HIDDEN_DIM
    mlp_layers: int = DEFAULT_TRAIN_MLP_LAYERS
    dropout: float = DEFAULT_TRAIN_DROPOUT
    position_embedding_type: PositionEmbeddingType = DEFAULT_POSITION_EMBEDDING_TYPE
    vocabulary_size: int | None = None
    initial_weights_path: str | None = None
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    max_checkpoint_files: int | None = DEFAULT_TRAIN_MAX_CHECKPOINT_FILES
    resume_checkpoint_path: str | None = None
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    auto_micro_batch: bool = False
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING_ENABLED
    tokenizer_path: str | None = None
    wandb_project: str | None = None
    tensorboard_dir: str | None = None


@dataclass(frozen=True)
class EpochMetric:
    """One epoch metric row.

    Attributes:
        epoch: One-based epoch index.
        train_loss: Average training loss.
        validation_loss: Average validation loss.
    """

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class BatchLossMetric:
    """One training batch metric row.

    Attributes:
        epoch: One-based epoch index.
        batch_index: One-based batch index inside the epoch.
        global_step: One-based global optimizer step index.
        train_loss: Batch training loss.
    """

    epoch: int
    batch_index: int
    global_step: int
    train_loss: float


@dataclass(frozen=True)
class TrainingRunResult:
    """Training command output artifact paths and summary metadata."""

    model_path: str
    history_path: str
    plot_path: str | None
    epochs_completed: int
    checkpoint_dir: str | None = None
    best_checkpoint_path: str | None = None
    resumed_from_checkpoint: str | None = None
    run_id: str | None = None
    artifact_contract_path: str | None = None
