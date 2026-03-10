"""Distributed train command wiring for Crucible CLI.

This module isolates the distributed-train subcommand parser and execution
logic. It launches torchrun as a subprocess to coordinate multi-GPU DDP
training across available GPUs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from core.errors import CrucibleDistributedError
from store.dataset_sdk import CrucibleClient


def add_distributed_train_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register distributed-train subcommand.

    Args:
        subparsers: Argparse subparsers group.
    """
    parser = subparsers.add_parser(
        "distributed-train",
        help="Launch multi-GPU DDP training via torchrun",
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Training artifact output directory",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=0,
        help="Number of GPUs per node (0 = auto-detect)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per GPU",
    )
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        help="Master address for distributed rendezvous",
    )
    parser.add_argument(
        "--master-port",
        default="29500",
        help="Master port for distributed rendezvous",
    )


def run_distributed_train_command(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Handle distributed-train command invocation.

    Launches torchrun with the DDP training entry point.

    Args:
        client: SDK client (used to resolve data root).
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    nproc = _resolve_gpu_count(args.nproc_per_node)
    entry_script = _resolve_entry_script()
    cmd = _build_torchrun_command(
        nproc=nproc,
        entry_script=str(entry_script),
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    print(f"Launching DDP training with {nproc} processes...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _resolve_gpu_count(requested: int) -> int:
    """Resolve GPU count, auto-detecting if zero.

    Args:
        requested: User-requested GPU count (0 = auto).

    Returns:
        Number of GPUs to use.
    """
    if requested > 0:
        return requested
    try:
        import torch
        count = torch.cuda.device_count()
        return max(count, 1)
    except ImportError:
        return 1


def _resolve_entry_script() -> Path:
    """Resolve path to the DDP entry point script."""
    return Path(__file__).parent.parent / "serve" / "ddp_entry.py"


def _build_torchrun_command(
    nproc: int,
    entry_script: str,
    dataset: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    master_addr: str,
    master_port: str,
) -> list[str]:
    """Build the torchrun command line.

    Returns:
        List of command arguments for subprocess.run.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        entry_script,
        "--dataset",
        dataset,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
    ]
    return cmd
