"""Unit tests for elastic fault-tolerant training."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from serve.fault_tolerant_training import (
    FaultTolerantConfig,
    build_elastic_launch_args,
    launch_elastic_training,
)


def test_fault_tolerant_config_frozen() -> None:
    """FaultTolerantConfig should be immutable (frozen dataclass)."""
    config = FaultTolerantConfig(
        min_nodes=1,
        max_nodes=4,
        master_addr="10.0.0.1",
        master_port=29500,
        nproc_per_node=2,
    )
    assert config.min_nodes == 1
    assert config.max_nodes == 4
    assert config.max_restarts == 3

    with pytest.raises(FrozenInstanceError):
        config.min_nodes = 5  # type: ignore[misc]


def test_build_elastic_args_includes_min_max() -> None:
    """build_elastic_launch_args should include min:max nnodes format."""
    args = build_elastic_launch_args(
        min_nodes=2,
        max_nodes=4,
        master_addr="10.0.0.1",
        master_port=29500,
        nproc_per_node=8,
        script_path="/path/to/train.py",
    )
    assert args[0] == sys.executable
    assert args[1] == "-m"
    assert args[2] == "torch.distributed.run"
    assert "--nnodes=2:4" in args
    assert "--nproc_per_node=8" in args
    assert "--master_addr=10.0.0.1" in args
    assert "--master_port=29500" in args
    assert args[-1] == "/path/to/train.py"


def test_build_elastic_args_with_defaults() -> None:
    """build_elastic_launch_args should work with min == max (fixed size)."""
    args = build_elastic_launch_args(
        min_nodes=1,
        max_nodes=1,
        master_addr="localhost",
        master_port=29400,
        nproc_per_node=1,
        script_path="/train.py",
    )
    assert "--nnodes=1:1" in args
    assert "--nproc_per_node=1" in args
    assert "--master_addr=localhost" in args
    assert "--master_port=29400" in args


@patch("serve.fault_tolerant_training.subprocess.run")
def test_launch_elastic_returns_exit_code(mock_run: MagicMock) -> None:
    """launch_elastic_training should return the subprocess exit code."""
    mock_run.return_value = MagicMock(returncode=0)

    exit_code = launch_elastic_training(
        min_nodes=1,
        max_nodes=3,
        master_addr="10.0.0.1",
        master_port=29500,
        nproc_per_node=2,
        script_path="/train.py",
        script_args=["--dataset", "test_ds"],
    )

    assert exit_code == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "--nnodes=1:3" in cmd
    assert "--dataset" in cmd
    assert "test_ds" in cmd
