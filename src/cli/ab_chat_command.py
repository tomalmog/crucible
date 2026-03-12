"""A/B chat comparison command wiring for Crucible CLI."""

from __future__ import annotations

import argparse

from serve.ab_chat import AbComparison, export_preferences_as_dpo, generate_ab_responses
from store.dataset_sdk import CrucibleClient


def run_ab_chat_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle ab-chat command invocation."""
    comparisons: list[AbComparison] = []
    comparison = generate_ab_responses(args.prompt, args.model_a, args.model_b)
    comparisons.append(comparison)
    print(f"prompt={args.prompt}")
    print(f"response_a={comparison.response_a}")
    print(f"response_b={comparison.response_b}")
    if args.export_dpo:
        count = export_preferences_as_dpo(comparisons, args.export_dpo)
        print(f"exported_dpo_pairs={count}")
    return 0


def add_ab_chat_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ab-chat subcommand."""
    parser = subparsers.add_parser("ab-chat", help="A/B model comparison chat")
    parser.add_argument("--model-a", required=True, help="Path to model A")
    parser.add_argument("--model-b", required=True, help="Path to model B")
    parser.add_argument("--prompt", required=True, help="User prompt to send to both models")
    parser.add_argument("--export-dpo", help="Export preferences as DPO data to this path")
