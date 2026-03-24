"""CLI command for SAE training."""

from __future__ import annotations

import argparse
from typing import Any


def add_sae_train_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("sae-train", help="Train a sparse autoencoder on layer activations")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--latent-dim", type=int, default=0, help="0 = auto (4x hidden)")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sparsity-coeff", type=float, default=1e-3)


def run_sae_train_command(client: Any, args: Any) -> int:
    from core.sae_types import SaeTrainOptions
    from serve.sae_train_runner import run_sae_train

    options = SaeTrainOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        base_model=getattr(args, "base_model", None),
        layer_index=args.layer_index,
        latent_dim=args.latent_dim,
        max_samples=args.max_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sparsity_coeff=args.sparsity_coeff,
    )
    dataset = client.dataset(options.dataset_name)
    _, records = dataset.load_records()
    run_sae_train(options, records)
    return 0
