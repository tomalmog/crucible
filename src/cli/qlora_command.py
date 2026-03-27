"""QLoRA command wiring for Crucible CLI.

This module isolates QLoRA command parser and execution logic,
mapping CLI arguments to QloraOptions for quantized LoRA training.
"""

from __future__ import annotations

import argparse
from typing import cast

from cli.training_output import print_and_register
from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
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
from core.qlora_types import (
    DEFAULT_QLORA_BITS,
    DEFAULT_QLORA_DOUBLE_QUANTIZE,
    DEFAULT_QLORA_TYPE,
    QloraOptions,
)
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_qlora_command(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Handle QLoRA command invocation."""
    target_modules = (
        tuple(args.lora_target_modules.split(","))
        if args.lora_target_modules
        else ("q_proj", "v_proj")
    )
    options = QloraOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        qlora_data_path=args.qlora_data_path,
        base_model_path=args.base_model_path,
        quantization_bits=args.quantization_bits,
        qlora_type=args.qlora_type,
        double_quantize=args.double_quantize,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
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
    result = client.qlora_train(options)
    print_and_register(client, result, args.model_name)
    return 0


def add_qlora_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register QLoRA subcommand."""
    parser = subparsers.add_parser(
        "qlora-train", help="Quantized LoRA fine-tuning",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory",
    )
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument(
        "--qlora-data-path",
        default="",
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="Path to base model to quantize",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=DEFAULT_QLORA_BITS,
        choices=[4, 8],
        help="Quantization bits",
    )
    parser.add_argument(
        "--qlora-type",
        default=DEFAULT_QLORA_TYPE,
        choices=["nf4", "fp4"],
        help="Quantization type",
    )
    parser.add_argument(
        "--no-double-quantize",
        action="store_false",
        dest="double_quantize",
        help="Disable double quantization",
    )
    parser.set_defaults(double_quantize=DEFAULT_QLORA_DOUBLE_QUANTIZE)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help="LoRA dropout rate",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated target modules",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAIN_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Max token length",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Validation fraction",
    )
    parser.add_argument(
        "--precision-mode",
        default=DEFAULT_TRAIN_PRECISION_MODE,
        choices=SUPPORTED_TRAIN_PRECISION_MODES,
        help="Precision mode",
    )
    parser.add_argument(
        "--optimizer-type",
        default=DEFAULT_TRAIN_OPTIMIZER_TYPE,
        choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES,
        help="Optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_TRAIN_WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_TRAIN_HIDDEN_DIM,
        help="Hidden dim",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_TRAIN_NUM_LAYERS,
        help="Num layers",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=DEFAULT_TRAIN_ATTENTION_HEADS,
        help="Attention heads",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=DEFAULT_TRAIN_MLP_HIDDEN_DIM,
        help="MLP hidden width",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        default=DEFAULT_TRAIN_MLP_LAYERS,
        help="MLP layers before vocab projection",
    )
    parser.add_argument("--hooks-file", help="Hook module path")
    parser.add_argument("--initial-weights-path", help="Initial weights")
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        action="store_false",
        dest="save_best_checkpoint",
        help="Disable best checkpoint",
    )
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument(
        "--progress-log-interval-steps",
        type=int,
        default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
        help="Log interval",
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
