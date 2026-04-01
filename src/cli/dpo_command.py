"""DPO command wiring for Crucible CLI.

This module isolates DPO command parser and execution logic,
mapping CLI arguments to DpoOptions for preference optimization.
"""

from __future__ import annotations

import argparse
from typing import cast

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DPO_BETA,
    DEFAULT_DPO_LABEL_SMOOTHING,
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
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
)
from cli.training_output import print_and_register
from core.dpo_types import DpoOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_dpo_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle DPO command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    options = DpoOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        dpo_data_path=args.dpo_data_path,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        reference_model_path=args.reference_model_path,
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
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        hooks_path=args.hooks_file,
        initial_weights_path=args.initial_weights_path,
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
    )
    result = client.dpo_train(options)
    print_and_register(client, result, args.model_name)
    return 0


def add_dpo_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register DPO subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "dpo-train",
        help="Direct Preference Optimization on preference pairs",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Training artifact output directory",
    )
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument(
        "--dpo-data-path", default="",
        help="Path to JSONL file with prompt/chosen/rejected triples",
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_DPO_BETA,
        help="DPO temperature parameter",
    )
    parser.add_argument(
        "--label-smoothing", type=float,
        default=DEFAULT_DPO_LABEL_SMOOTHING,
        help="Label smoothing factor for robustness",
    )
    parser.add_argument(
        "--reference-model-path",
        help="Optional path to pre-trained reference model weights",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS,
        help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=5e-5, help="Optimizer learning rate",
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
        "--mlp-hidden-dim", type=int,
        default=DEFAULT_TRAIN_MLP_HIDDEN_DIM,
        help="MLP hidden width",
    )
    parser.add_argument(
        "--mlp-layers", type=int,
        default=DEFAULT_TRAIN_MLP_LAYERS,
        help="MLP layers before vocab projection",
    )
    parser.add_argument(
        "--hooks-file",
        help="Optional .py hook module with callback functions",
    )
    parser.add_argument(
        "--initial-weights-path",
        help="Optional model artifact used as initial weights",
    )
    parser.add_argument("--base-model", help="HuggingFace model ID (e.g. 'gpt2') to use as base architecture")
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
    parser.add_argument(
        "--tokenizer-path", default=None,
        help="Path to tokenizer or HuggingFace tokenizer ID",
    )
    parser.add_argument(
        "--resume-checkpoint-path", default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Name for model registry (auto-derived if not set)",
    )
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project name for experiment tracking",
    )
    parser.add_argument(
        "--tensorboard-dir", default=None,
        help="TensorBoard log directory for experiment tracking",
    )
