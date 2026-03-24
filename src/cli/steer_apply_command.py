"""CLI command for applying a steering vector."""

from __future__ import annotations

import argparse
from typing import Any


def add_steer_apply_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("steer-apply", help="Generate text with a steering vector applied")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steering-vector-path", required=True, help="Path to steering vector .pt file")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--coefficient", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=50)


def run_steer_apply_command(client: Any, args: Any) -> int:
    from core.steering_types import SteerApplyOptions
    from serve.steer_apply_runner import run_steer_apply

    options = SteerApplyOptions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        steering_vector_path=args.steering_vector_path,
        input_text=args.input_text,
        base_model=getattr(args, "base_model", None),
        coefficient=args.coefficient,
        max_new_tokens=args.max_new_tokens,
    )
    run_steer_apply(options)
    return 0
