"""ORPO training typed models.

This module defines immutable data models for Odds Ratio Preference
Optimization combining SFT and preference learning in a single step.
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

DEFAULT_ORPO_LAMBDA = 1.0
DEFAULT_ORPO_BETA = 0.1


@dataclass(frozen=True)
class OrpoExample:
    """One ORPO preference pair for combined SFT + preference training.

    Attributes:
        prompt: User prompt text.
        chosen: Preferred model response.
        rejected: Non-preferred model response.
    """

    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class OrpoOptions:
    """ORPO training configuration used by CLI, SDK, and run-spec workflows.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        orpo_data_path: Path to JSONL with preference pairs.
    """

    dataset_name: str
    output_dir: str
    orpo_data_path: str = ""
    lambda_orpo: float = DEFAULT_ORPO_LAMBDA
    beta: float = DEFAULT_ORPO_BETA
    version_id: str | None = None
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
