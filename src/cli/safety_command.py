"""Safety CLI command wiring for Forge.

This module registers the safety-eval and safety-gate subcommands,
mapping CLI arguments to safety evaluation and gating functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.errors import ForgeSafetyError
from core.safety_types import SafetyEvalConfig
from safety.safety_gate import format_gate_result, run_safety_gate
from safety.toxicity_scorer import score_batch_toxicity
from store.dataset_sdk import ForgeClient


def _load_eval_texts(eval_data_path: str) -> list[str]:
    """Load evaluation texts from a JSON file.

    Args:
        eval_data_path: Path to JSON file with text samples.

    Returns:
        List of text strings.

    Raises:
        ForgeSafetyError: If the file cannot be loaded.
    """
    path = Path(eval_data_path)
    if not path.exists():
        raise ForgeSafetyError(f"Eval data not found: {eval_data_path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ForgeSafetyError(f"Failed to load eval data: {exc}") from exc
    if not isinstance(data, list):
        raise ForgeSafetyError("Eval data must be a JSON array of strings.")
    return [str(item) for item in data]


def run_safety_eval_command(
    client: ForgeClient,
    args: argparse.Namespace,
) -> int:
    """Handle safety-eval command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    texts = _load_eval_texts(args.eval_data)
    config = SafetyEvalConfig(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
    )
    scores = score_batch_toxicity(texts, threshold=config.toxicity_threshold)
    for ts in scores:
        flag = " [FLAGGED]" if ts.flagged else ""
        print(f"  score={ts.score:.4f}{flag}  {ts.text[:80]}")
    flagged = sum(1 for s in scores if s.flagged)
    print(f"\nTotal: {len(scores)}, Flagged: {flagged}")
    return 0


def add_safety_eval_command(subparsers: Any) -> None:
    """Register safety-eval subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "safety-eval",
        help="Run safety toxicity evaluation on model outputs",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained model weights file",
    )
    parser.add_argument(
        "--eval-data", required=True,
        help="Path to JSON file with evaluation texts",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for safety report output",
    )


def run_safety_gate_command(
    client: ForgeClient,
    args: argparse.Namespace,
) -> int:
    """Handle safety-gate command invocation.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code (0 if passed, 1 if failed).
    """
    texts = _load_eval_texts(args.eval_data)
    config = SafetyEvalConfig(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        toxicity_threshold=args.threshold,
    )
    result = run_safety_gate(config, texts)
    for line in format_gate_result(result):
        print(line)
    return 0 if result.passed else 1


def add_safety_gate_command(subparsers: Any) -> None:
    """Register safety-gate subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "safety-gate",
        help="Run pre-deploy safety gate check",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained model weights file",
    )
    parser.add_argument(
        "--eval-data", required=True,
        help="Path to JSON file with evaluation texts",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for safety report output",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Toxicity threshold for gate (default 0.5)",
    )
