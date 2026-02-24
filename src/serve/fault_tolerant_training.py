"""Elastic training with automatic restart on node failure.

This module provides fault-tolerant training configuration and launch
functions using torchrun elastic launch mode. Elastic training allows
the cluster to continue when nodes join or leave, automatically
restarting failed workers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from core.errors import ForgeDistributedError

_DEFAULT_MAX_RESTARTS = 3


@dataclass(frozen=True)
class FaultTolerantConfig:
    """Configuration for elastic fault-tolerant training.

    Attributes:
        min_nodes: Minimum number of nodes to start training.
        max_nodes: Maximum number of nodes in the elastic group.
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        nproc_per_node: Number of processes (GPUs) per node.
        max_restarts: Maximum number of worker restarts on failure.
    """

    min_nodes: int
    max_nodes: int
    master_addr: str
    master_port: int
    nproc_per_node: int
    max_restarts: int = _DEFAULT_MAX_RESTARTS


def build_elastic_launch_args(
    min_nodes: int,
    max_nodes: int,
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    script_path: str,
) -> list[str]:
    """Build torchrun elastic launch command-line arguments.

    Args:
        min_nodes: Minimum nodes required to start training.
        max_nodes: Maximum nodes in the elastic group.
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        nproc_per_node: Number of processes (GPUs) per node.
        script_path: Path to the training entry script.

    Returns:
        List of command-line arguments for subprocess.run.

    Raises:
        ForgeDistributedError: If configuration is invalid.
    """
    _validate_elastic_config(min_nodes, max_nodes, nproc_per_node)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={min_nodes}:{max_nodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path,
    ]


def launch_elastic_training(
    min_nodes: int,
    max_nodes: int,
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    script_path: str,
    script_args: list[str],
) -> int:
    """Launch elastic fault-tolerant training via torchrun.

    Args:
        min_nodes: Minimum nodes required to start training.
        max_nodes: Maximum nodes in the elastic group.
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        nproc_per_node: Number of processes (GPUs) per node.
        script_path: Path to the training entry script.
        script_args: Additional arguments for the training script.

    Returns:
        Exit code from the torchrun subprocess.

    Raises:
        ForgeDistributedError: If configuration is invalid or launch fails.
    """
    cmd = build_elastic_launch_args(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        master_addr=master_addr,
        master_port=master_port,
        nproc_per_node=nproc_per_node,
        script_path=script_path,
    )
    cmd.extend(script_args)

    env = os.environ.copy()
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)

    try:
        result = subprocess.run(cmd, env=env, check=False)
    except OSError as error:
        raise ForgeDistributedError(
            f"Failed to launch elastic training subprocess: {error}"
        ) from error
    return result.returncode


def _validate_elastic_config(
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
) -> None:
    """Validate elastic training parameters.

    Args:
        min_nodes: Minimum number of nodes.
        max_nodes: Maximum number of nodes.
        nproc_per_node: Processes per node.

    Raises:
        ForgeDistributedError: If any parameter is invalid.
    """
    if min_nodes < 1:
        raise ForgeDistributedError(
            f"min_nodes must be at least 1, got {min_nodes}."
        )
    if max_nodes < min_nodes:
        raise ForgeDistributedError(
            f"max_nodes ({max_nodes}) must be >= min_nodes ({min_nodes})."
        )
    if nproc_per_node < 1:
        raise ForgeDistributedError(
            f"nproc_per_node must be at least 1, got {nproc_per_node}."
        )
