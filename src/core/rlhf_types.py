"""RLHF training typed models.

This module defines immutable data models for Reinforcement Learning from
Human Feedback training configuration including PPO hyperparameters and
reward model settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DPO_BETA,
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


@dataclass(frozen=True)
class PpoConfig:
    """PPO hyperparameters for policy optimization.

    Attributes:
        clip_epsilon: PPO clipping range for probability ratio.
        value_loss_coeff: Coefficient for value function loss term.
        entropy_coeff: Coefficient for entropy bonus term.
        ppo_epochs: Number of PPO optimization passes per batch.
        gamma: Discount factor for future rewards.
        lam: GAE lambda for advantage estimation.
    """

    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95


@dataclass(frozen=True)
class RewardModelConfig:
    """Reward model configuration for RLHF training.

    Attributes:
        reward_model_path: Path to external pre-trained reward model.
        train_reward_model: Whether to train reward model from preference data.
        preference_data_path: Path to JSONL with prompt/chosen/rejected triples.
    """

    reward_model_path: str | None = None
    train_reward_model: bool = False
    preference_data_path: str | None = None


@dataclass(frozen=True)
class RlhfOptions:
    """RLHF training configuration used by CLI, SDK, and run-spec workflows.

    Attributes:
        dataset_name: Logical dataset identifier.
        output_dir: Training artifact output directory.
        policy_model_path: Path to the policy model checkpoint.
    """

    dataset_name: str
    output_dir: str
    policy_model_path: str
    reward_config: RewardModelConfig = RewardModelConfig()
    ppo_config: PpoConfig = PpoConfig()
    version_id: str | None = None
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = 1e-5
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
