"""Crucible CLI entry points.
This module maps argparse commands onto SDK calls via a dispatch table.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from cli.ab_chat_command import add_ab_chat_command, run_ab_chat_command
from cli.activation_pca_command import add_activation_pca_command, run_activation_pca_command
from cli.activation_patching_command import (
    add_activation_patching_command,
    run_activation_patching_command,
)
from cli.benchmark_command import add_benchmark_command, run_benchmark_command
from cli.chat_command import add_chat_command, run_chat_command
from cli.compare_command import add_compare_command, run_compare_command
from cli.compute_command import add_compute_command, run_compute_command
from cli.curate_command import add_curate_command, run_curate_command
from cli.data_commands import (
    add_dataset_command,
    add_export_training_command,
    add_ingest_command,
    run_dataset_command,
    run_export_training_command,
    run_ingest_command,
)
from cli.dispatch_command import add_dispatch_command, run_dispatch_command
from cli.distillation_command import add_distillation_command, run_distillation_command
from cli.job_command import add_job_command, run_job_command
from cli.distributed_train_command import (
    add_distributed_train_command,
    run_distributed_train_command,
)
from cli.domain_adapt_command import add_domain_adapt_command, run_domain_adapt_command
from cli.dpo_command import add_dpo_command, run_dpo_command
from cli.eval_command import add_eval_command, run_eval_command
from cli.export_spec_command import add_export_spec_command, run_export_spec_command
from cli.grpo_command import add_grpo_command, run_grpo_command
from cli.hardware_profile_command import (
    add_hardware_profile_command,
    run_hardware_profile_command,
)
from cli.hub_command import add_hub_command, run_hub_command
from cli.kto_command import add_kto_command, run_kto_command
from cli.linear_probe_command import add_linear_probe_command, run_linear_probe_command
from cli.logit_lens_command import add_logit_lens_command, run_logit_lens_command
from cli.lora_command import (
    add_lora_merge_command,
    add_lora_train_command,
    run_lora_merge_command,
    run_lora_train_command,
)
from cli.merge_command import add_merge_command, run_merge_command
from cli.onnx_export_command import add_onnx_export_command, run_onnx_export_command
from cli.safetensors_export_command import (
    add_safetensors_export_command,
    run_safetensors_export_command,
)
from cli.gguf_export_command import add_gguf_export_command, run_gguf_export_command
from cli.hf_export_command import add_hf_export_command, run_hf_export_command
from cli.model_command import add_model_command, run_model_command
from cli.multimodal_command import add_multimodal_command, run_multimodal_command
from cli.orpo_command import add_orpo_command, run_orpo_command
from cli.qlora_command import add_qlora_command, run_qlora_command
from cli.recipe_command import add_recipe_command, run_recipe_command
from cli.remote_command import add_remote_command, run_remote_command
from cli.replay_command import add_replay_command, run_replay_command
from cli.rlhf_command import add_rlhf_command, run_rlhf_command
from cli.rlvr_command import add_rlvr_command, run_rlvr_command
from cli.sae_analyze_command import add_sae_analyze_command, run_sae_analyze_command
from cli.sae_train_command import add_sae_train_command, run_sae_train_command
from cli.run_spec_command import add_run_spec_command, run_run_spec_command
from cli.server_command import add_server_command, run_server_command
from cli.sft_command import add_sft_command, run_sft_command
from cli.steer_apply_command import add_steer_apply_command, run_steer_apply_command
from cli.steer_compute_command import add_steer_compute_command, run_steer_compute_command
from cli.suggest_command import add_suggest_command, run_suggest_command
from cli.sweep_command import add_sweep_command, run_sweep_command
from cli.synthetic_command import add_synthetic_command, run_synthetic_command
from cli.train_command import add_train_command, run_train_command
from cli.verify_command import add_verify_command, run_verify_command
from core.config import CrucibleConfig
from serve.training_progress import emit_progress
from store.dataset_sdk import CrucibleClient

_CommandHandler = Callable[..., int]

_COMMAND_REGISTRARS: tuple[Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None], ...] = (
    add_ingest_command, add_dataset_command,
    add_export_training_command, add_run_spec_command, add_verify_command,
    add_hardware_profile_command, add_train_command, add_sft_command,
    add_dpo_command, add_distillation_command, add_domain_adapt_command,
    add_lora_train_command, add_lora_merge_command, add_export_spec_command,
    add_chat_command, add_rlhf_command, add_distributed_train_command,
    add_benchmark_command, add_sweep_command, add_compare_command,
    add_replay_command, add_model_command,
    add_compute_command, add_server_command,
    add_grpo_command, add_qlora_command, add_kto_command, add_orpo_command,
    add_suggest_command, add_hub_command,
    add_eval_command, add_curate_command,
    add_merge_command, add_ab_chat_command, add_recipe_command,
    add_multimodal_command,
    add_synthetic_command, add_rlvr_command, add_remote_command,
    add_logit_lens_command, add_activation_pca_command, add_activation_patching_command,
    add_linear_probe_command, add_sae_train_command, add_sae_analyze_command,
    add_steer_compute_command, add_steer_apply_command,
    add_dispatch_command, add_job_command,
    add_onnx_export_command,
    add_safetensors_export_command, add_gguf_export_command,
    add_hf_export_command,
)


def _build_dispatch_table() -> dict[str, _CommandHandler]:
    """Build command name -> handler mapping."""
    return {
        "ingest": run_ingest_command,
        "dataset": run_dataset_command,
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
        "compute": run_compute_command,
        "server": run_server_command,
        "grpo-train": run_grpo_command,
        "qlora-train": run_qlora_command,
        "kto-train": run_kto_command,
        "orpo-train": run_orpo_command,
        "suggest": run_suggest_command,
        "hub": run_hub_command,
        "eval": run_eval_command,
        "curate": run_curate_command,
        "merge": run_merge_command,
        "ab-chat": run_ab_chat_command,
        "recipe": run_recipe_command,
        "multimodal-train": run_multimodal_command,
        "synthetic": run_synthetic_command,
        "rlvr-train": run_rlvr_command,
        "remote": run_remote_command,
        "logit-lens": run_logit_lens_command,
        "activation-pca": run_activation_pca_command,
        "activation-patch": run_activation_patching_command,
        "linear-probe": run_linear_probe_command,
        "sae-train": run_sae_train_command,
        "sae-analyze": run_sae_analyze_command,
        "steer-compute": run_steer_compute_command,
        "steer-apply": run_steer_apply_command,
        "dispatch": run_dispatch_command,
        "job": run_job_command,
        "onnx-export": run_onnx_export_command,
        "safetensors-export": run_safetensors_export_command,
        "gguf-export": run_gguf_export_command,
        "hf-export": run_hf_export_command,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="crucible", description="Crucible CLI")
    parser.add_argument("--data-root", help="Override CRUCIBLE_DATA_ROOT for this command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for registrar in _COMMAND_REGISTRARS:
        registrar(subparsers)
    subparsers.add_parser("mcp-server", help="Start the Crucible MCP server for Claude Code integration")
    return parser


_TRAINING_COMMANDS = frozenset({
    "train", "sft", "dpo-train", "rlhf-train", "lora-train", "distill",
    "domain-adapt", "grpo-train", "qlora-train", "kto-train", "orpo-train",
    "multimodal-train", "rlvr-train", "distributed-train",
})


def _init_backends() -> None:
    """Register all execution backends."""
    from core.backend_registry import register_backend
    from serve.local_runner import LocalRunner
    from serve.slurm_runner import SlurmRunner
    from serve.ssh_runner import SshRunner
    from serve.http_api_runner import HttpApiRunner

    register_backend("local", LocalRunner())
    register_backend("slurm", SlurmRunner())
    register_backend("ssh", SshRunner())
    register_backend("http-api", HttpApiRunner())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Crucible CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # MCP server runs standalone — no client or backends needed
    if args.command == "mcp-server":
        from serve.mcp_server import run_mcp_server
        run_mcp_server()
        return 0
    _init_backends()
    client = _build_client(args.data_root)
    dispatch = _build_dispatch_table()
    handler = dispatch.get(args.command)
    if handler is None:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    if args.command in _TRAINING_COMMANDS:
        emit_progress("training_preparing", method=args.command)
    return handler(client, args)


def _build_client(data_root: str | None) -> CrucibleClient:
    """Build SDK client with optional data-root override."""
    config = CrucibleConfig.from_env()
    if data_root:
        config = replace(config, data_root=Path(data_root).expanduser().resolve())
    return CrucibleClient(config)
