"""Multi-node training launcher coordinating torchrun across machines.

This module provides functions to launch torchrun subprocesses configured
for multi-node distributed training, and to build script arguments for
the training entry point.
"""

from __future__ import annotations

import os
import subprocess

from core.errors import ForgeDistributedError
from serve.multinode_setup import (
    build_multinode_env,
    build_torchrun_multinode_args,
)


def build_multinode_script_args(
    dataset: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
) -> list[str]:
    """Build script arguments for the multi-node training script.

    Args:
        dataset: Name of the dataset to train on.
        output_dir: Directory for training artifacts.
        epochs: Number of training epochs.
        batch_size: Batch size per GPU.

    Returns:
        List of script argument strings.
    """
    return [
        "--dataset",
        dataset,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]


def launch_multinode_training(
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    num_nodes: int,
    node_rank: int,
    script_path: str,
    script_args: list[str],
) -> int:
    """Launch torchrun subprocess for multi-node distributed training.

    Builds the torchrun command with multi-node arguments, sets the
    required environment variables, and runs the subprocess.

    Args:
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        nproc_per_node: Number of processes (GPUs) per node.
        num_nodes: Total number of nodes.
        node_rank: Rank of this node.
        script_path: Path to the training entry script.
        script_args: Additional arguments for the training script.

    Returns:
        Exit code from the torchrun subprocess.

    Raises:
        ForgeDistributedError: If configuration is invalid or launch fails.
    """
    cmd = build_torchrun_multinode_args(
        master_addr=master_addr,
        master_port=master_port,
        nproc_per_node=nproc_per_node,
        num_nodes=num_nodes,
        node_rank=node_rank,
        script_path=script_path,
    )
    cmd.extend(script_args)

    env = _build_subprocess_env(
        master_addr=master_addr,
        master_port=master_port,
        node_rank=node_rank,
        num_nodes=num_nodes,
    )

    try:
        result = subprocess.run(cmd, env=env, check=False)
    except OSError as error:
        raise ForgeDistributedError(
            f"Failed to launch multi-node training subprocess: {error}"
        ) from error
    return result.returncode


def _build_subprocess_env(
    master_addr: str,
    master_port: int,
    node_rank: int,
    num_nodes: int,
) -> dict[str, str]:
    """Merge multi-node env vars with the current environment.

    Args:
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        node_rank: Rank of this node.
        num_nodes: Total number of nodes.

    Returns:
        Full environment dict for subprocess.run.
    """
    env = os.environ.copy()
    multinode_vars = build_multinode_env(
        master_addr=master_addr,
        master_port=master_port,
        node_rank=node_rank,
        num_nodes=num_nodes,
    )
    env.update(multinode_vars)
    return env
