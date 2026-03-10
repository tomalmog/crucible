"""Suggest command wiring for Crucible CLI.

This module provides hardware-aware training configuration suggestions.
"""

from __future__ import annotations

import argparse

from serve.gpu_profiles import list_gpu_profiles
from serve.smart_config import suggest_training_config
from store.dataset_sdk import CrucibleClient


def _parse_model_size(value: str) -> float:
    """Parse model size string into billions of parameters.

    Accepts: 7, 0.125, 125M, 1.5B, 7b, 350m
    """
    v = value.strip().lower()
    if v.endswith("b"):
        return float(v[:-1])
    if v.endswith("m"):
        return float(v[:-1]) / 1000
    return float(v)


def run_suggest_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle suggest command invocation."""
    if args.list_gpus:
        profiles = list_gpu_profiles()
        print("Available GPU profiles:")
        for p in profiles:
            print(f"  {p.name}: {p.vram_gb}GB VRAM, {p.fp16_tflops} FP16 TFLOPS")
        return 0
    suggestion = suggest_training_config(
        model_size_billions=args.model_size,
        training_method=args.method,
        gpu_name=args.gpu,
        dataset_size_examples=args.dataset_size,
        target_epochs=args.epochs,
    )
    print(f"gpu={suggestion.gpu_name}")
    print(f"batch_size={suggestion.batch_size}")
    print(f"precision_mode={suggestion.precision_mode}")
    print(f"gradient_accumulation_steps={suggestion.gradient_accumulation_steps}")
    print(f"use_qlora={suggestion.use_qlora}")
    print(f"estimated_memory_gb={suggestion.estimated_memory_gb}")
    print(f"estimated_time_hours={suggestion.estimated_time_hours}")
    for note in suggestion.notes:
        print(f"note={note}")
    return 0


def add_suggest_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register suggest subcommand."""
    parser = subparsers.add_parser("suggest", help="Get hardware-aware training config suggestions")
    parser.add_argument("--model-size", type=_parse_model_size, default=7.0, help="Model size in billions of parameters (e.g. 7, 0.125, 125M, 1.5B)")
    parser.add_argument("--method", default="sft", help="Training method (sft, dpo-train, etc.)")
    parser.add_argument("--gpu", default="rtx4090", help="GPU model name")
    parser.add_argument("--dataset-size", type=int, default=10000, help="Number of training examples")
    parser.add_argument("--epochs", type=int, default=3, help="Target training epochs")
    parser.add_argument("--list-gpus", action="store_true", help="List known GPU profiles")
