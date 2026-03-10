"""Multi-node rendezvous configuration for distributed training.

This module builds environment variables and torchrun command-line arguments
needed to launch multi-node distributed training across multiple machines.
"""

from __future__ import annotations

import sys

from core.errors import CrucibleDistributedError

_MIN_PORT = 1
_MAX_PORT = 65535
_MIN_NODES = 1


def validate_multinode_config(
    master_addr: str,
    master_port: int,
    num_nodes: int,
) -> None:
    """Validate multi-node configuration parameters.

    Args:
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        num_nodes: Total number of nodes in the cluster.

    Raises:
        CrucibleDistributedError: If any parameter is invalid.
    """
    if not master_addr or not master_addr.strip():
        raise CrucibleDistributedError(
            "master_addr must be a non-empty string."
        )
    if master_port < _MIN_PORT or master_port > _MAX_PORT:
        raise CrucibleDistributedError(
            f"master_port must be between {_MIN_PORT} and {_MAX_PORT}, "
            f"got {master_port}."
        )
    if num_nodes < _MIN_NODES:
        raise CrucibleDistributedError(
            f"num_nodes must be at least {_MIN_NODES}, got {num_nodes}."
        )


def build_multinode_env(
    master_addr: str,
    master_port: int,
    node_rank: int,
    num_nodes: int,
) -> dict[str, str]:
    """Build environment variables dict for torchrun multi-node launch.

    These variables are read by torch.distributed.run to coordinate
    rendezvous across nodes.

    Args:
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        node_rank: Rank of this node (0-indexed).
        num_nodes: Total number of nodes in the cluster.

    Returns:
        Dictionary of environment variable name-value pairs.

    Raises:
        CrucibleDistributedError: If configuration is invalid.
    """
    validate_multinode_config(master_addr, master_port, num_nodes)
    if node_rank < 0 or node_rank >= num_nodes:
        raise CrucibleDistributedError(
            f"node_rank must be between 0 and {num_nodes - 1}, "
            f"got {node_rank}."
        )
    return {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "NNODES": str(num_nodes),
        "NODE_RANK": str(node_rank),
    }


def build_torchrun_multinode_args(
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    num_nodes: int,
    node_rank: int,
    script_path: str,
) -> list[str]:
    """Build full torchrun command-line args for multi-node training.

    Args:
        master_addr: IP or hostname of the master node.
        master_port: Port for distributed rendezvous.
        nproc_per_node: Number of processes (GPUs) per node.
        num_nodes: Total number of nodes.
        node_rank: Rank of this node.
        script_path: Path to the training script.

    Returns:
        List of command-line arguments for subprocess.run.

    Raises:
        CrucibleDistributedError: If configuration is invalid.
    """
    validate_multinode_config(master_addr, master_port, num_nodes)
    if nproc_per_node < 1:
        raise CrucibleDistributedError(
            f"nproc_per_node must be at least 1, got {nproc_per_node}."
        )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path,
    ]
