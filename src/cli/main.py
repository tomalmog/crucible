"""Forge CLI entry points.
This module maps argparse commands onto SDK calls via a dispatch table.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from cli.benchmark_command import add_benchmark_command, run_benchmark_command
from cli.chat_command import add_chat_command, run_chat_command
from cli.compare_command import add_compare_command, run_compare_command
from cli.compute_command import add_compute_command, run_compute_command
from cli.deploy_command import add_deploy_command, run_deploy_command
from cli.data_commands import (
    add_export_training_command,
    add_filter_command,
    add_ingest_command,
    add_versions_command,
    run_export_training_command,
    run_filter_command,
    run_ingest_command,
    run_versions_command,
)
from cli.distillation_command import add_distillation_command, run_distillation_command
from cli.distributed_train_command import (
    add_distributed_train_command,
    run_distributed_train_command,
)
from cli.domain_adapt_command import add_domain_adapt_command, run_domain_adapt_command
from cli.dpo_command import add_dpo_command, run_dpo_command
from cli.export_spec_command import add_export_spec_command, run_export_spec_command
from cli.hardware_profile_command import (
    add_hardware_profile_command,
    run_hardware_profile_command,
)
from cli.lora_command import (
    add_lora_merge_command,
    add_lora_train_command,
    run_lora_merge_command,
    run_lora_train_command,
)
from cli.model_command import add_model_command, run_model_command
from cli.replay_command import add_replay_command, run_replay_command
from cli.rlhf_command import add_rlhf_command, run_rlhf_command
from cli.run_spec_command import add_run_spec_command, run_run_spec_command
from cli.server_command import add_server_command, run_server_command
from cli.safety_command import (
    add_safety_eval_command,
    add_safety_gate_command,
    run_safety_eval_command,
    run_safety_gate_command,
)
from cli.sft_command import add_sft_command, run_sft_command
from cli.sweep_command import add_sweep_command, run_sweep_command
from cli.train_command import add_train_command, run_train_command
from cli.verify_command import add_verify_command, run_verify_command
from core.config import ForgeConfig
from store.dataset_sdk import ForgeClient

_CommandHandler = Callable[..., int]

_COMMAND_REGISTRARS: tuple[Callable[[Any], None], ...] = (
    add_ingest_command, add_versions_command, add_filter_command,
    add_export_training_command, add_run_spec_command, add_verify_command,
    add_hardware_profile_command, add_train_command, add_sft_command,
    add_dpo_command, add_distillation_command, add_domain_adapt_command,
    add_lora_train_command, add_lora_merge_command, add_export_spec_command,
    add_chat_command, add_rlhf_command, add_distributed_train_command,
    add_benchmark_command, add_sweep_command, add_compare_command,
    add_replay_command, add_model_command,
    add_safety_eval_command, add_safety_gate_command,
    add_compute_command, add_server_command,
    add_deploy_command,
)


def _build_dispatch_table() -> dict[str, _CommandHandler]:
    """Build command name -> handler mapping."""
    return {
        "ingest": run_ingest_command,
        "versions": run_versions_command,
        "filter": run_filter_command,
        "export-training": run_export_training_command,
        "train": run_train_command,
        "sft": run_sft_command,
        "dpo-train": run_dpo_command,
        "distill": run_distillation_command,
        "domain-adapt": run_domain_adapt_command,
        "lora-train": run_lora_train_command,
        "lora-merge": lambda client, args: run_lora_merge_command(args),
        "export-spec": run_export_spec_command,
        "chat": run_chat_command,
        "run-spec": run_run_spec_command,
        "verify": run_verify_command,
        "hardware-profile": lambda client, args: run_hardware_profile_command(),
        "rlhf-train": run_rlhf_command,
        "distributed-train": run_distributed_train_command,
        "benchmark": run_benchmark_command,
        "sweep": run_sweep_command,
        "compare": run_compare_command,
        "replay": run_replay_command,
        "model": run_model_command,
        "safety-eval": run_safety_eval_command,
        "safety-gate": run_safety_gate_command,
        "compute": run_compute_command,
        "server": run_server_command,
        "deploy": run_deploy_command,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="forge", description="Forge CLI")
    parser.add_argument("--data-root", help="Override FORGE_DATA_ROOT for this command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for registrar in _COMMAND_REGISTRARS:
        registrar(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Forge CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    client = _build_client(args.data_root)
    dispatch = _build_dispatch_table()
    handler = dispatch.get(args.command)
    if handler is None:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    return handler(client, args)


def _build_client(data_root: str | None) -> ForgeClient:
    """Build SDK client with optional data-root override."""
    config = ForgeConfig.from_env()
    if data_root:
        config = replace(config, data_root=Path(data_root).expanduser().resolve())
    return ForgeClient(config)
