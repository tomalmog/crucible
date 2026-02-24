"""Unit tests for multi-node training launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from serve.multinode_launcher import (
    build_multinode_script_args,
    launch_multinode_training,
)


def test_build_script_args_format() -> None:
    """build_multinode_script_args should produce correct flag-value pairs."""
    args = build_multinode_script_args(
        dataset="my_dataset",
        output_dir="/out/training",
        epochs=5,
        batch_size=16,
    )
    assert args == [
        "--dataset",
        "my_dataset",
        "--output-dir",
        "/out/training",
        "--epochs",
        "5",
        "--batch-size",
        "16",
    ]


@patch("serve.multinode_launcher.subprocess.run")
def test_launch_returns_exit_code(mock_run: MagicMock) -> None:
    """launch_multinode_training should return the subprocess exit code."""
    mock_run.return_value = MagicMock(returncode=0)

    exit_code = launch_multinode_training(
        master_addr="10.0.0.1",
        master_port=29500,
        nproc_per_node=2,
        num_nodes=2,
        node_rank=0,
        script_path="/path/to/train.py",
        script_args=["--dataset", "ds1"],
    )

    assert exit_code == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "/path/to/train.py" in cmd
    assert "--dataset" in cmd
    assert "ds1" in cmd


@patch("serve.multinode_launcher.subprocess.run")
def test_launch_with_extra_args(mock_run: MagicMock) -> None:
    """launch_multinode_training should append extra script args."""
    mock_run.return_value = MagicMock(returncode=1)

    exit_code = launch_multinode_training(
        master_addr="192.168.1.1",
        master_port=30000,
        nproc_per_node=4,
        num_nodes=3,
        node_rank=2,
        script_path="/entry.py",
        script_args=["--epochs", "10", "--batch-size", "32"],
    )

    assert exit_code == 1
    cmd = mock_run.call_args[0][0]
    assert "--epochs" in cmd
    assert "10" in cmd
    assert "--batch-size" in cmd
    assert "32" in cmd
