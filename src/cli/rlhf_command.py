"""RLHF command wiring for Forge CLI.

This module isolates RLHF command parser and execution logic,
mapping CLI arguments to RlhfOptions for reinforcement learning
from human feedback training.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
)
from core.rlhf_types import PpoConfig, RewardModelConfig, RlhfOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import ForgeClient


def run_rlhf_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle RLHF command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    reward_config = RewardModelConfig(
        reward_model_path=args.reward_model_path,
        train_reward_model=args.train_reward_model,
        preference_data_path=args.preference_data_path,
    )
    ppo_config = PpoConfig(
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        entropy_coeff=args.entropy_coeff,
    )
    options = RlhfOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        policy_model_path=args.policy_model_path,
        reward_config=reward_config,
        ppo_config=ppo_config,
        version_id=args.version_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length,
        validation_split=args.validation_split,
        precision_mode=cast(PrecisionMode, args.precision_mode),
        optimizer_type=cast(OptimizerType, args.optimizer_type),
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        hooks_path=args.hooks_file,
        initial_weights_path=args.initial_weights_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
    )
    result = client.rlhf_train(options)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"run_id={result.run_id or '-'}")
    print(f"artifact_contract_path={result.artifact_contract_path or '-'}")
    return 0


def add_rlhf_command(subparsers: Any) -> None:
    """Register RLHF subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "rlhf-train",
        help="RLHF training with PPO and reward model",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--output-dir", required=True,
        help="Training artifact output directory",
    )
    parser.add_argument(
        "--policy-model-path", required=True,
        help="Path to policy model checkpoint",
    )
    parser.add_argument(
        "--reward-model-path",
        help="Path to external pre-trained reward model",
    )
    parser.add_argument(
        "--train-reward-model", action="store_true",
        help="Train reward model from preference data",
    )
    parser.add_argument(
        "--preference-data-path",
        help="JSONL file with prompt/chosen/rejected for reward training",
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2,
        help="PPO clipping epsilon",
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4,
        help="PPO optimization passes per batch",
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.01,
        help="Entropy bonus coefficient",
    )
    parser.add_argument("--version-id", help="Optional specific version id")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS,
        help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=DEFAULT_TRAIN_LEARNING_RATE, help="Optimizer learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Maximum token length per sequence",
    )
    parser.add_argument(
        "--validation-split", type=float,
        default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Validation data fraction in [0,1)",
    )
    parser.add_argument(
        "--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE,
        choices=SUPPORTED_TRAIN_PRECISION_MODES,
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE,
        choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES,
        help="Optimizer backend",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM,
        help="Default model hidden size",
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS,
        help="Default model layer count",
    )
    parser.add_argument(
        "--attention-heads", type=int,
        default=DEFAULT_TRAIN_ATTENTION_HEADS,
        help="Attention heads per transformer layer",
    )
    parser.add_argument(
        "--hooks-file",
        help="Optional .py hook module with callback functions",
    )
    parser.add_argument(
        "--initial-weights-path",
        help="Optional model artifact used as initial weights",
    )
    parser.add_argument(
        "--checkpoint-every-epochs", type=int,
        default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        help="Save training checkpoint every N epochs",
    )
    parser.add_argument(
        "--no-save-best-checkpoint", action="store_false",
        dest="save_best_checkpoint",
        help="Disable writing best.pt checkpoint",
    )
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument(
        "--progress-log-interval-steps", type=int,
        default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
        help="Log training batch progress every N batches",
    )
