"""Distillation command wiring for Forge CLI.

This module isolates distillation command parser and execution logic,
mapping CLI arguments to DistillationOptions for knowledge distillation.
"""

from __future__ import annotations

import argparse
from typing import cast

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTILLATION_ALPHA,
    DEFAULT_DISTILLATION_TEMPERATURE,
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
from core.distillation_types import DistillationOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import ForgeClient


def run_distillation_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle distillation command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    options = DistillationOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        teacher_model_path=args.teacher_model_path,
        student_model_path=args.student_model_path,
        temperature=args.temperature,
        alpha=args.alpha,
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
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        hooks_path=args.hooks_file,
        initial_weights_path=args.initial_weights_path,
        tokenizer_path=args.tokenizer_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
    )
    result = client.distill(options)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"run_id={result.run_id or '-'}")
    print(f"artifact_contract_path={result.artifact_contract_path or '-'}")
    return 0


def add_distillation_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register distill subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "distill",
        help="Knowledge distillation from teacher to student model",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--output-dir", required=True, help="Training artifact output directory",
    )
    parser.add_argument(
        "--teacher-model-path", required=True, help="Path to teacher model weights",
    )
    parser.add_argument(
        "--student-model-path", default=None, help="Path to student model weights",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_DISTILLATION_TEMPERATURE,
        help="Softening temperature for KL divergence",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_DISTILLATION_ALPHA,
        help="Blend coefficient: alpha*KL + (1-alpha)*CE",
    )
    parser.add_argument("--version-id", help="Optional specific version id")
    _add_shared_training_args(parser)


def _add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    """Add standard training arguments shared with other commands."""
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size",
    )
    parser.add_argument(
        "--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Maximum token length per sequence",
    )
    parser.add_argument(
        "--validation-split", type=float, default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Validation data fraction in [0,1)",
    )
    parser.add_argument(
        "--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE,
        choices=SUPPORTED_TRAIN_PRECISION_MODES, help="Mixed precision mode",
    )
    parser.add_argument(
        "--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE,
        choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES, help="Optimizer backend",
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
        "--attention-heads", type=int, default=DEFAULT_TRAIN_ATTENTION_HEADS,
        help="Attention heads per transformer layer",
    )
    parser.add_argument(
        "--mlp-hidden-dim", type=int, default=DEFAULT_TRAIN_MLP_HIDDEN_DIM,
        help="Hidden width of transformer feed-forward block",
    )
    parser.add_argument(
        "--mlp-layers", type=int, default=DEFAULT_TRAIN_MLP_LAYERS,
        help="Number of MLP layers before vocabulary projection",
    )
    parser.add_argument(
        "--hooks-file", help="Optional .py hook module with callback functions",
    )
    parser.add_argument(
        "--initial-weights-path", help="Optional model artifact for initial weights",
    )
    parser.add_argument(
        "--checkpoint-every-epochs", type=int,
        default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        help="Save training checkpoint every N epochs",
    )
    parser.add_argument(
        "--no-save-best-checkpoint", action="store_false",
        dest="save_best_checkpoint", help="Disable writing best.pt checkpoint",
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
