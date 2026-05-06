"""Unit tests for remote chat Slurm runtime commands."""

from __future__ import annotations

from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.remote_chat_runner import _build_srun_command


def _make_resources() -> SlurmResourceConfig:
    return SlurmResourceConfig(
        partition="gpu",
        nodes=1,
        gpus_per_node=1,
        gpu_type="a100",
        cpus_per_task=4,
        memory="32G",
        time_limit="00:30:00",
    )


def test_build_srun_command_uses_managed_env_python() -> None:
    """Managed env chat runs should target the activated conda interpreter."""
    cluster = ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="testuser",
        default_partition="gpu",
        module_loads=("module load cuda/12.1",),
        python_path="python3",
    )
    command = _build_srun_command(cluster, "/scratch/chat-bundle", _make_resources())
    expected = (
        '"${CONDA_PREFIX:?managed crucible env inactive}/bin/python" '
        "-u _chat_runner.py"
    )
    assert expected in command


def test_build_srun_command_uses_explicit_python_path() -> None:
    """Explicit chat runtimes should keep their configured interpreter."""
    cluster = ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="testuser",
        default_partition="gpu",
        python_path="/shared/envs/demo/bin/python",
    )
    command = _build_srun_command(cluster, "/scratch/chat-bundle", _make_resources())
    assert "/shared/envs/demo/bin/python -u _chat_runner.py" in command


def test_build_srun_command_includes_excluded_nodes() -> None:
    """Chat runtime should pass cluster excluded nodes through to srun."""
    cluster = ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="testuser",
        default_partition="gpu",
        python_path="/shared/envs/demo/bin/python",
        exclude_nodes="node-a,node-b",
    )
    command = _build_srun_command(cluster, "/scratch/chat-bundle", _make_resources())
    assert "--exclude=node-a,node-b" in command
