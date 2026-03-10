"""LLM-as-Judge command wiring for Crucible CLI.

This module provides the judge command for evaluating model
outputs using an external LLM judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.llm_judge import DEFAULT_CRITERIA, JudgeCriteria, LlmJudge
from store.dataset_sdk import CrucibleClient


def run_judge_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle judge command invocation."""
    criteria_names = args.criteria.split(",") if args.criteria else None
    if criteria_names:
        criteria = tuple(
            c for c in DEFAULT_CRITERIA if c.name in criteria_names
        )
    else:
        criteria = DEFAULT_CRITERIA
    judge = LlmJudge(
        judge_api_url=args.judge_api,
        api_key=args.api_key or "",
        criteria=criteria,
    )
    test_prompts = _load_test_prompts(args.test_prompts)
    responses = ["(placeholder response)"] * len(test_prompts)
    result = judge.evaluate_model(
        model_path=args.model_path,
        test_prompts=test_prompts,
        responses=responses,
    )
    print(f"model_path={result.model_path}")
    print(f"average_score={result.average_score}")
    print(f"num_prompts={result.num_prompts}")
    for s in result.scores:
        print(f"criteria={s.criteria}  score={s.score}  explanation={s.explanation}")
    return 0


def _load_test_prompts(path: str) -> list[str]:
    """Load test prompts from a JSONL file."""
    prompts: list[str] = []
    p = Path(path)
    if not p.exists():
        return ["Hello, how are you?"]
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj.get("prompt", line))
    return prompts


def add_judge_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register judge subcommand."""
    parser = subparsers.add_parser("judge", help="Evaluate model with LLM-as-Judge")
    parser.add_argument("--model-path", required=True, help="Path to model to evaluate")
    parser.add_argument("--judge-api", required=True, help="Judge API endpoint URL")
    parser.add_argument("--api-key", help="API key for the judge service")
    parser.add_argument(
        "--criteria",
        default=None,
        help="Comma-separated criteria (helpfulness,accuracy,safety,reasoning)",
    )
    parser.add_argument(
        "--test-prompts",
        default="test_prompts.jsonl",
        help="Path to JSONL file with test prompts",
    )
