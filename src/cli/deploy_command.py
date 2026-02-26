"""Deploy command wiring for Forge CLI.

This module provides sub-subcommands under ``forge deploy`` for
packaging, quantization, latency profiling, and readiness checking.
"""

from __future__ import annotations

import argparse

from core.deployment_types import QuantizationConfig
from deploy.latency_profiler import (
    format_latency_report,
    profile_model_latency,
)
from deploy.packaging import PackageConfig, build_deployment_package
from deploy.quantization_pipeline import run_quantization
from deploy.readiness_checklist import (
    format_checklist,
    run_readiness_checklist,
)
from store.dataset_sdk import ForgeClient


def add_deploy_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register deploy subcommand with its sub-subcommands.

    Args:
        subparsers: Argparse subparsers object from the top-level parser.
    """
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deployment tooling commands",
    )
    deploy_sub = deploy_parser.add_subparsers(
        dest="deploy_action", required=True,
    )

    _add_package_subcommand(deploy_sub)
    _add_quantize_subcommand(deploy_sub)
    _add_profile_subcommand(deploy_sub)
    _add_checklist_subcommand(deploy_sub)


def run_deploy_command(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    """Dispatch deploy sub-subcommand to the correct handler.

    Args:
        client: SDK client.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    handlers = {
        "package": _handle_package,
        "quantize": _handle_quantize,
        "profile": _handle_profile,
        "checklist": _handle_checklist,
    }
    handler = handlers.get(args.deploy_action)
    if handler is None:
        print(f"Unknown deploy action: {args.deploy_action}")
        return 2
    return handler(args)


def _add_package_subcommand(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'package' sub-subcommand."""
    p = sub.add_parser("package", help="Build deployment package")
    p.add_argument(
        "--model-path", required=True,
        help="Path to the model file",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for the package",
    )
    p.add_argument("--config-path", default=None, help="Config file")
    p.add_argument(
        "--tokenizer-path", default=None, help="Tokenizer file",
    )
    p.add_argument(
        "--safety-report-path", default=None, help="Safety report",
    )


def _add_quantize_subcommand(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'quantize' sub-subcommand."""
    p = sub.add_parser("quantize", help="Quantize ONNX model")
    p.add_argument(
        "--model-path", required=True,
        help="Path to the ONNX model file",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for quantized model",
    )
    p.add_argument(
        "--type", dest="quant_type", default="dynamic",
        choices=("dynamic", "static"),
        help="Quantization type",
    )


def _add_profile_subcommand(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'profile' sub-subcommand."""
    p = sub.add_parser("profile", help="Profile model latency")
    p.add_argument(
        "--model-path", required=True,
        help="Path to the ONNX model file",
    )
    p.add_argument(
        "--batch-sizes", default="1,4,8",
        help="Comma-separated batch sizes",
    )
    p.add_argument(
        "--seq-lengths", default="32,128,512",
        help="Comma-separated sequence lengths",
    )
    p.add_argument(
        "--device", default="cpu", help="Device for profiling",
    )
    p.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of runs per configuration",
    )


def _add_checklist_subcommand(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'checklist' sub-subcommand."""
    p = sub.add_parser(
        "checklist", help="Run deployment readiness checklist",
    )
    p.add_argument(
        "--model-path", required=True,
        help="Path to the model file",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory containing deployment artifacts",
    )


def _handle_package(args: argparse.Namespace) -> int:
    """Execute the package sub-subcommand."""
    config = PackageConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        safety_report_path=args.safety_report_path,
    )
    pkg = build_deployment_package(config)
    print(f"Package built: {pkg.package_path}")
    print(f"Model checksum: {pkg.checksum}")
    return 0


def _handle_quantize(args: argparse.Namespace) -> int:
    """Execute the quantize sub-subcommand."""
    config = QuantizationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization_type=args.quant_type,
    )
    output_path = run_quantization(config)
    print(f"Quantized model written to: {output_path}")
    return 0


def _handle_profile(args: argparse.Namespace) -> int:
    """Execute the profile sub-subcommand."""
    batch_sizes = tuple(
        int(s) for s in args.batch_sizes.split(",")
    )
    seq_lengths = tuple(
        int(s) for s in args.seq_lengths.split(",")
    )
    profiles = profile_model_latency(
        model_path=args.model_path,
        batch_sizes=batch_sizes,
        sequence_lengths=seq_lengths,
        device=args.device,
        num_runs=args.num_runs,
    )
    report = format_latency_report(profiles)
    for line in report:
        print(line)
    return 0


def _handle_checklist(args: argparse.Namespace) -> int:
    """Execute the checklist sub-subcommand."""
    checklist = run_readiness_checklist(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    report = format_checklist(checklist)
    for line in report:
        print(line)
    return 0 if checklist.all_passed else 1
