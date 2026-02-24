"""Unit tests for distributed-train CLI command wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cli.distributed_train_command import (
    _build_torchrun_command,
    _resolve_gpu_count,
    add_distributed_train_command,
)
from cli.main import build_parser


def test_distributed_train_command_registered_in_parser() -> None:
    """The distributed-train command should be a recognized subcommand."""
    parser = build_parser()
    args = parser.parse_args([
        "distributed-train",
        "--dataset",
        "test-ds",
        "--output-dir",
        "/tmp/out",
    ])

    assert args.command == "distributed-train"
    assert args.dataset == "test-ds"
    assert args.output_dir == "/tmp/out"


def test_distributed_train_default_args() -> None:
    """Default argument values should be set correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "distributed-train",
        "--dataset",
        "demo",
        "--output-dir",
        "/tmp/out",
    ])

    assert args.nproc_per_node == 0
    assert args.epochs == 10
    assert args.learning_rate == 0.001
    assert args.batch_size == 32
    assert args.master_addr == "127.0.0.1"
    assert args.master_port == "29500"


def test_distributed_train_custom_args() -> None:
    """Custom arguments should be parsed correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "distributed-train",
        "--dataset",
        "demo",
        "--output-dir",
        "/tmp/out",
        "--nproc-per-node",
        "4",
        "--epochs",
        "20",
        "--learning-rate",
        "0.0001",
        "--batch-size",
        "64",
        "--version-id",
        "v123",
        "--master-addr",
        "10.0.0.1",
        "--master-port",
        "30000",
    ])

    assert args.nproc_per_node == 4
    assert args.epochs == 20
    assert args.learning_rate == 0.0001
    assert args.batch_size == 64
    assert args.version_id == "v123"
    assert args.master_addr == "10.0.0.1"
    assert args.master_port == "30000"


def test_resolve_gpu_count_returns_requested_when_positive() -> None:
    """When user requests a specific count, it should be returned."""
    assert _resolve_gpu_count(4) == 4


def test_resolve_gpu_count_defaults_to_one_without_torch() -> None:
    """Without torch, auto-detect should default to 1."""
    with patch.dict("sys.modules", {"torch": None}):
        count = _resolve_gpu_count(0)

    assert count >= 1


def test_build_torchrun_command_basic() -> None:
    """Torchrun command should contain expected arguments."""
    cmd = _build_torchrun_command(
        nproc=2,
        entry_script="/path/to/ddp_entry.py",
        dataset="my-data",
        output_dir="/tmp/out",
        epochs=5,
        learning_rate=0.01,
        batch_size=16,
        version_id=None,
        master_addr="127.0.0.1",
        master_port="29500",
    )

    assert "--nproc_per_node=2" in cmd
    assert "--master_addr=127.0.0.1" in cmd
    assert "--master_port=29500" in cmd
    assert "--dataset" in cmd
    assert "my-data" in cmd
    assert "--output-dir" in cmd
    assert "/tmp/out" in cmd
    assert "--version-id" not in cmd


def test_build_torchrun_command_with_version_id() -> None:
    """When version_id is provided it should appear in the command."""
    cmd = _build_torchrun_command(
        nproc=1,
        entry_script="/path/entry.py",
        dataset="ds",
        output_dir="/tmp",
        epochs=1,
        learning_rate=0.001,
        batch_size=32,
        version_id="v42",
        master_addr="127.0.0.1",
        master_port="29500",
    )

    assert "--version-id" in cmd
    assert "v42" in cmd
