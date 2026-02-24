"""Unit tests for multi-node rendezvous configuration."""

from __future__ import annotations

import sys

import pytest

from core.errors import ForgeDistributedError
from serve.multinode_setup import (
    build_multinode_env,
    build_torchrun_multinode_args,
    validate_multinode_config,
)


def test_build_env_includes_all_keys() -> None:
    """build_multinode_env should return all required env var keys."""
    env = build_multinode_env(
        master_addr="10.0.0.1",
        master_port=29500,
        node_rank=0,
        num_nodes=4,
    )
    assert env["MASTER_ADDR"] == "10.0.0.1"
    assert env["MASTER_PORT"] == "29500"
    assert env["NNODES"] == "4"
    assert env["NODE_RANK"] == "0"
    assert len(env) == 4


def test_validate_valid_config() -> None:
    """validate_multinode_config should not raise for valid params."""
    validate_multinode_config(
        master_addr="192.168.1.10",
        master_port=29500,
        num_nodes=2,
    )


def test_validate_invalid_port_raises() -> None:
    """validate_multinode_config should raise for out-of-range ports."""
    with pytest.raises(ForgeDistributedError, match="master_port"):
        validate_multinode_config(
            master_addr="10.0.0.1",
            master_port=0,
            num_nodes=2,
        )
    with pytest.raises(ForgeDistributedError, match="master_port"):
        validate_multinode_config(
            master_addr="10.0.0.1",
            master_port=70000,
            num_nodes=2,
        )


def test_validate_zero_nodes_raises() -> None:
    """validate_multinode_config should raise when num_nodes < 1."""
    with pytest.raises(ForgeDistributedError, match="num_nodes"):
        validate_multinode_config(
            master_addr="10.0.0.1",
            master_port=29500,
            num_nodes=0,
        )


def test_build_torchrun_args_format() -> None:
    """build_torchrun_multinode_args should produce correct torchrun args."""
    args = build_torchrun_multinode_args(
        master_addr="10.0.0.1",
        master_port=29500,
        nproc_per_node=4,
        num_nodes=2,
        node_rank=1,
        script_path="/path/to/train.py",
    )
    assert args[0] == sys.executable
    assert args[1] == "-m"
    assert args[2] == "torch.distributed.run"
    assert "--nnodes=2" in args
    assert "--node_rank=1" in args
    assert "--nproc_per_node=4" in args
    assert "--master_addr=10.0.0.1" in args
    assert "--master_port=29500" in args
    assert args[-1] == "/path/to/train.py"
