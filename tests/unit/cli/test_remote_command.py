"""Unit tests for remote CLI command parsing."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_remote_register_cluster_parses() -> None:
    """register-cluster should parse required and optional args."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "register-cluster",
        "--name", "my-hpc",
        "--host", "hpc.example.com",
        "--user", "jdoe",
        "--ssh-key", "/home/jdoe/.ssh/id_rsa",
        "--partition", "gpu",
        "--module-loads", "module load cuda/12.1,module load python/3.11",
        "--remote-workspace", "/scratch/forge",
    ])
    assert args.command == "remote"
    assert args.remote_action == "register-cluster"
    assert args.name == "my-hpc"
    assert args.host == "hpc.example.com"
    assert args.user == "jdoe"
    assert args.ssh_key == "/home/jdoe/.ssh/id_rsa"
    assert args.partition == "gpu"
    assert args.module_loads == "module load cuda/12.1,module load python/3.11"
    assert args.remote_workspace == "/scratch/forge"


def test_remote_register_cluster_requires_name() -> None:
    """register-cluster should fail without --name."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "remote", "register-cluster",
            "--host", "h", "--user", "u",
        ])


def test_remote_list_clusters_parses() -> None:
    """list-clusters should parse with no extra args."""
    parser = build_parser()
    args = parser.parse_args(["remote", "list-clusters"])
    assert args.command == "remote"
    assert args.remote_action == "list-clusters"


def test_remote_validate_cluster_parses() -> None:
    """validate-cluster should parse --cluster."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "validate-cluster", "--cluster", "my-hpc",
    ])
    assert args.remote_action == "validate-cluster"
    assert args.cluster == "my-hpc"


def test_remote_remove_cluster_parses() -> None:
    """remove-cluster should parse --cluster."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "remove-cluster", "--cluster", "my-hpc",
    ])
    assert args.remote_action == "remove-cluster"
    assert args.cluster == "my-hpc"


def test_remote_submit_parses() -> None:
    """submit should parse cluster, method, and resource args."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "submit",
        "--cluster", "my-hpc",
        "--method", "sft",
        "--method-args", '{"sft_data_path": "/data/sft.jsonl"}',
        "--partition", "gpu",
        "--nodes", "2",
        "--gpus-per-node", "4",
        "--gpu-type", "a100",
        "--cpus-per-task", "8",
        "--memory", "64G",
        "--time-limit", "24:00:00",
        "--pull-model",
    ])
    assert args.remote_action == "submit"
    assert args.cluster == "my-hpc"
    assert args.method == "sft"
    assert args.method_args == '{"sft_data_path": "/data/sft.jsonl"}'
    assert args.partition == "gpu"
    assert args.nodes == 2
    assert args.gpus_per_node == 4
    assert args.gpu_type == "a100"
    assert args.cpus_per_task == 8
    assert args.memory == "64G"
    assert args.time_limit == "24:00:00"
    assert args.pull_model is True


def test_remote_submit_with_model_name() -> None:
    """submit should parse --model-name."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "submit",
        "--cluster", "c",
        "--method", "train",
        "--method-args", "{}",
        "--model-name", "My-Transformer",
    ])
    assert args.model_name == "My-Transformer"


def test_remote_submit_defaults() -> None:
    """submit should have sensible defaults for optional fields."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "submit",
        "--cluster", "c",
        "--method", "train",
        "--method-args", "{}",
    ])
    assert args.nodes == 1
    assert args.gpus_per_node == 1
    assert args.cpus_per_task == 4
    assert args.memory == "32G"
    assert args.time_limit == "12:00:00"
    assert args.pull_model is False
    assert args.partition == ""
    assert args.gpu_type == ""
    assert args.model_name == ""


def test_remote_submit_requires_cluster() -> None:
    """submit should fail without --cluster."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "remote", "submit",
            "--method", "sft",
            "--method-args", "{}",
        ])


def test_remote_submit_sweep_parses() -> None:
    """submit-sweep should parse sweep config path."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "submit-sweep",
        "--cluster", "my-hpc",
        "--method", "sft",
        "--sweep-config", "sweep.yaml",
    ])
    assert args.remote_action == "submit-sweep"
    assert args.sweep_config == "sweep.yaml"


def test_remote_list_parses() -> None:
    """list should parse with no extra args."""
    parser = build_parser()
    args = parser.parse_args(["remote", "list"])
    assert args.remote_action == "list"


def test_remote_status_parses() -> None:
    """status should parse --job-id."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "status", "--job-id", "rj-abc123def456",
    ])
    assert args.remote_action == "status"
    assert args.job_id == "rj-abc123def456"


def test_remote_logs_parses() -> None:
    """logs should parse --job-id, --follow, and --tail."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "logs",
        "--job-id", "rj-abc123",
        "--follow",
        "--tail", "200",
    ])
    assert args.remote_action == "logs"
    assert args.job_id == "rj-abc123"
    assert args.follow is True
    assert args.tail == 200


def test_remote_logs_defaults() -> None:
    """logs should default to no-follow and 100 tail lines."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "logs", "--job-id", "rj-abc",
    ])
    assert args.follow is False
    assert args.tail == 100


def test_remote_cancel_parses() -> None:
    """cancel should parse --job-id."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "cancel", "--job-id", "rj-abc123",
    ])
    assert args.remote_action == "cancel"
    assert args.job_id == "rj-abc123"


def test_remote_pull_model_parses() -> None:
    """pull-model should parse --job-id and --model-name."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "pull-model",
        "--job-id", "rj-abc123",
        "--model-name", "my-sft-model",
    ])
    assert args.remote_action == "pull-model"
    assert args.job_id == "rj-abc123"
    assert args.model_name == "my-sft-model"


def test_remote_pull_model_name_optional() -> None:
    """pull-model --model-name should be optional."""
    parser = build_parser()
    args = parser.parse_args([
        "remote", "pull-model", "--job-id", "rj-abc123",
    ])
    assert args.model_name is None


def test_remote_requires_action() -> None:
    """remote without subcommand should fail."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["remote"])
