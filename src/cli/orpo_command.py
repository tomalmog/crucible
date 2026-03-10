"""ORPO command wiring for Crucible CLI.

This module isolates ORPO command parser and execution logic,
mapping CLI arguments to OrpoOptions for odds ratio preference optimization.
"""

from __future__ import annotations

import argparse
from typing import cast

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
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
)
from core.orpo_types import DEFAULT_ORPO_BETA, DEFAULT_ORPO_LAMBDA, OrpoOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_orpo_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle ORPO command invocation."""
    options = OrpoOptions(
        dataset_name=args.dataset, output_dir=args.output_dir,
        orpo_data_path=args.orpo_data_path,
        lambda_orpo=args.lambda_orpo, beta=args.beta,
        epochs=args.epochs,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        max_token_length=args.max_token_length, validation_split=args.validation_split,
        precision_mode=cast(PrecisionMode, args.precision_mode),
        optimizer_type=cast(OptimizerType, args.optimizer_type),
        weight_decay=args.weight_decay, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, attention_heads=args.attention_heads,
        mlp_hidden_dim=args.mlp_hidden_dim, mlp_layers=args.mlp_layers,
        hooks_path=args.hooks_file, initial_weights_path=args.initial_weights_path,
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
    )
    result = client.orpo_train(options)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"run_id={result.run_id or '-'}")
    print(f"artifact_contract_path={result.artifact_contract_path or '-'}")
    return 0


def add_orpo_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ORPO subcommand."""
    parser = subparsers.add_parser("orpo-train", help="Odds Ratio Preference Optimization training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument("--orpo-data-path", default="", help="Path to JSONL with preference pairs")
    parser.add_argument("--lambda-orpo", type=float, default=DEFAULT_ORPO_LAMBDA, help="ORPO lambda weight for odds-ratio term")
    parser.add_argument("--beta", type=float, default=DEFAULT_ORPO_BETA, help="ORPO beta parameter")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH, help="Max token length")
    parser.add_argument("--validation-split", type=float, default=DEFAULT_TRAIN_VALIDATION_SPLIT, help="Validation fraction")
    parser.add_argument("--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE, choices=SUPPORTED_TRAIN_PRECISION_MODES, help="Precision mode")
    parser.add_argument("--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE, choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES, help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM, help="Hidden dim")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS, help="Num layers")
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_TRAIN_ATTENTION_HEADS, help="Attention heads")
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_TRAIN_MLP_HIDDEN_DIM, help="MLP hidden width")
    parser.add_argument("--mlp-layers", type=int, default=DEFAULT_TRAIN_MLP_LAYERS, help="MLP layers before vocab projection")
    parser.add_argument("--hooks-file", help="Hook module path")
    parser.add_argument("--initial-weights-path", help="Initial weights")
    parser.add_argument("--base-model", help="HuggingFace model ID (e.g. 'gpt2') to use as base architecture")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS, help="Checkpoint frequency")
    parser.add_argument("--no-save-best-checkpoint", action="store_false", dest="save_best_checkpoint", help="Disable best checkpoint")
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument("--progress-log-interval-steps", type=int, default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS, help="Log interval")
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer or HuggingFace tokenizer ID")
    parser.add_argument("--resume-checkpoint-path", default=None, help="Path to checkpoint to resume training from")
