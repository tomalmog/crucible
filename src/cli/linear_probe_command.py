"""CLI command for linear probe analysis."""

from __future__ import annotations

import argparse
from typing import Any


def add_linear_probe_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("linear-probe", help="Train linear probes on frozen activations")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-field", required=True, help="Metadata field for classification target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--layer-index", type=int, default=-1, help="-1=last, -2=all layers")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)


def run_linear_probe_command(client: Any, args: Any) -> int:
    from core.linear_probe_types import LinearProbeOptions
    from serve.linear_probe_runner import run_linear_probe

    options = LinearProbeOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        label_field=args.label_field,
        base_model=getattr(args, "base_model", None),
        layer_index=args.layer_index,
        max_samples=args.max_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    dataset = client.dataset(options.dataset_name)
    _, records = dataset.load_records()
    run_linear_probe(options, records)
    return 0
