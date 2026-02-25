"""Synthetic data generation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from serve.synthetic_data import export_synthetic_data, filter_by_quality, generate_synthetic_data
from store.dataset_sdk import ForgeClient


def run_synthetic_command(client: ForgeClient, args: argparse.Namespace) -> int:
    seed_prompts = _load_seed_prompts(args.seed_prompts)
    if not seed_prompts:
        print("No seed prompts found.")
        return 1
    examples = generate_synthetic_data(seed_prompts, args.count, args.model_path)
    filtered = filter_by_quality(examples, args.min_quality)
    count = export_synthetic_data(filtered, args.output)
    print(f"generated={len(examples)}")
    print(f"filtered={len(filtered)}")
    print(f"exported={count}")
    print(f"output_path={args.output}")
    return 0


def _load_seed_prompts(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    prompts = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", line))
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def add_synthetic_command(subparsers: Any) -> None:
    parser = subparsers.add_parser("synthetic", help="Generate synthetic training data")
    parser.add_argument("--model-path", help="Model to use for generation")
    parser.add_argument("--seed-prompts", required=True, help="Path to seed prompts file")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--min-quality", type=float, default=0.5, help="Minimum quality score")
    parser.add_argument("--output", default="./synthetic_data.jsonl", help="Output file path")
