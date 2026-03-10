"""RLVR training command wiring."""

from __future__ import annotations

import argparse
from typing import cast

from core.constants import *
from core.rlvr_types import RlvrOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_rlvr_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    options = RlvrOptions(
        dataset_name=args.dataset, output_dir=args.output_dir,
        rlvr_data_path=args.rlvr_data_path, verifier_type=args.verifier_type,
        max_verification_attempts=args.max_attempts,
        reward_correct=args.reward_correct, reward_incorrect=args.reward_incorrect,
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
    result = client.rlvr_train(options)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"run_id={result.run_id or '-'}")
    return 0


def add_rlvr_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("rlvr-train", help="RL with verifiable rewards training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument("--rlvr-data-path", default="", help="Path to verifiable tasks JSONL")
    parser.add_argument("--verifier-type", default="code", choices=["code", "math"], help="Verifier type")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max verification attempts")
    parser.add_argument("--reward-correct", type=float, default=1.0, help="Reward for correct")
    parser.add_argument("--reward-incorrect", type=float, default=-0.5, help="Reward for incorrect")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_TRAIN_VALIDATION_SPLIT)
    parser.add_argument("--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE, choices=SUPPORTED_TRAIN_PRECISION_MODES)
    parser.add_argument("--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE, choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_TRAIN_ATTENTION_HEADS)
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_TRAIN_MLP_HIDDEN_DIM)
    parser.add_argument("--mlp-layers", type=int, default=DEFAULT_TRAIN_MLP_LAYERS)
    parser.add_argument("--hooks-file", help="Hook module")
    parser.add_argument("--initial-weights-path", help="Initial weights")
    parser.add_argument("--base-model", help="HuggingFace model ID (e.g. 'gpt2') to use as base architecture")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument("--no-save-best-checkpoint", action="store_false", dest="save_best_checkpoint")
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument("--progress-log-interval-steps", type=int, default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS)
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer or HuggingFace tokenizer ID")
    parser.add_argument("--resume-checkpoint-path", default=None, help="Path to checkpoint to resume training from")
