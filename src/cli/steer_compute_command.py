"""CLI command for steering vector computation."""

from __future__ import annotations

import argparse
from typing import Any


def add_steer_compute_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("steer-compute", help="Compute a steering vector from contrastive examples")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-text", default="", help="Positive example text (simple mode)")
    parser.add_argument("--negative-text", default="", help="Negative example text (simple mode)")
    parser.add_argument("--positive-dataset", default="", help="Positive dataset name (dataset mode)")
    parser.add_argument("--negative-dataset", default="", help="Negative dataset name (dataset mode)")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=100)


def run_steer_compute_command(client: Any, args: Any) -> int:
    from core.steering_types import SteerComputeOptions
    from serve.steer_compute_runner import run_steer_compute

    options = SteerComputeOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        positive_text=args.positive_text,
        negative_text=args.negative_text,
        positive_dataset=args.positive_dataset,
        negative_dataset=args.negative_dataset,
        base_model=getattr(args, "base_model", None),
        layer_index=args.layer_index,
        max_samples=args.max_samples,
    )

    pos_records = None
    neg_records = None
    if options.positive_dataset and options.negative_dataset:
        _, pos_records = client.dataset(options.positive_dataset).load_records()
        _, neg_records = client.dataset(options.negative_dataset).load_records()

    run_steer_compute(options, pos_records, neg_records)
    return 0
