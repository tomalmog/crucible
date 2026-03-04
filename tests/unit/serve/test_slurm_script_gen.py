"""Unit tests for Slurm sbatch script generation."""

from __future__ import annotations

from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.slurm_script_gen import (
    generate_multi_node_script,
    generate_single_node_script,
    generate_sweep_script,
)


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="testuser",
        default_partition="gpu",
        module_loads=("module load cuda/12.1", "module load python/3.11"),
        python_path="python3",
        remote_workspace="/scratch/forge",
    )


def _make_resources(**overrides: object) -> SlurmResourceConfig:
    defaults = {
        "partition": "gpu",
        "nodes": 1,
        "gpus_per_node": 4,
        "gpu_type": "a100",
        "cpus_per_task": 8,
        "memory": "64G",
        "time_limit": "24:00:00",
    }
    defaults.update(overrides)
    return SlurmResourceConfig(**defaults)  # type: ignore[arg-type]


def test_single_node_script_has_shebang() -> None:
    """Single-node script should start with #!/bin/bash."""
    script = generate_single_node_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft",
    )
    assert script.startswith("#!/bin/bash\n")


def test_single_node_script_sbatch_directives() -> None:
    """Single-node script should contain proper SBATCH directives."""
    script = generate_single_node_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft",
    )
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:a100:4" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --output=/scratch/forge/rj-test123/slurm-%j.out" in script


def test_single_node_script_job_name() -> None:
    """Job name should include the training method."""
    script = generate_single_node_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft",
    )
    assert "#SBATCH --job-name=forge-sft-" in script


def test_single_node_script_module_loads() -> None:
    """Script should include module load commands from cluster config."""
    script = generate_single_node_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft",
    )
    assert "module load cuda/12.1" in script
    assert "module load python/3.11" in script


def test_single_node_script_execution() -> None:
    """Script should cd, extract tarball, and run entry script."""
    script = generate_single_node_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft",
    )
    assert "cd /scratch/forge/rj-test123" in script
    assert "tar xzf forge-agent.tar.gz" in script
    assert "python3 forge_agent_entry.py --config training_config.json" in script


def test_single_node_script_no_partition_when_empty() -> None:
    """Omit partition directive when neither resource nor cluster has one."""
    cluster = ClusterConfig(name="c", host="h", user="u")
    resources = _make_resources(partition="")
    script = generate_single_node_script(cluster, resources, "rj-x", "train")
    assert "--partition" not in script


def test_single_node_gpu_type_omitted() -> None:
    """When gpu_type is empty, gres should just have the count."""
    resources = _make_resources(gpu_type="", gpus_per_node=2)
    script = generate_single_node_script(
        _make_cluster(), resources, "rj-test123", "sft",
    )
    assert "#SBATCH --gres=gpu:2" in script


def test_multi_node_script_has_torchrun() -> None:
    """Multi-node script should use srun + torch.distributed.run."""
    script = generate_multi_node_script(
        _make_cluster(), _make_resources(nodes=4), "rj-test123", "sft",
    )
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "srun python3 -m torch.distributed.run" in script


def test_multi_node_script_nccl_env() -> None:
    """Multi-node script should set NCCL and master env vars."""
    script = generate_multi_node_script(
        _make_cluster(), _make_resources(nodes=2), "rj-test123", "sft",
    )
    assert "export MASTER_ADDR=" in script
    assert "export MASTER_PORT=29500" in script
    assert "export NCCL_IB_DISABLE=0" in script
    assert "export NCCL_DEBUG=INFO" in script


def test_multi_node_script_torchrun_args() -> None:
    """Multi-node script should pass correct torchrun args."""
    script = generate_multi_node_script(
        _make_cluster(), _make_resources(nodes=4), "rj-test123", "sft",
    )
    assert "--nproc_per_node=$SLURM_GPUS_ON_NODE" in script
    assert "--nnodes=$SLURM_NNODES" in script
    assert "--node_rank=$SLURM_NODEID" in script
    assert "--master_addr=$MASTER_ADDR" in script
    assert "--master_port=$MASTER_PORT" in script


def test_sweep_script_has_array() -> None:
    """Sweep script should use --array directive."""
    script = generate_sweep_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft", array_size=20,
    )
    assert "#SBATCH --array=0-19" in script


def test_sweep_script_uses_task_id() -> None:
    """Sweep script should reference SLURM_ARRAY_TASK_ID for config."""
    script = generate_sweep_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft", array_size=5,
    )
    assert "trials/trial_${SLURM_ARRAY_TASK_ID}.json" in script


def test_sweep_script_output_format() -> None:
    """Sweep script output should use %A_%a for array job/task ID."""
    script = generate_sweep_script(
        _make_cluster(), _make_resources(), "rj-test123", "sft", array_size=3,
    )
    assert "slurm-%A_%a.out" in script


def test_extra_sbatch_directives() -> None:
    """Extra sbatch directives from SlurmResourceConfig should appear."""
    resources = _make_resources(extra_sbatch=(("account", "myproject"),))
    script = generate_single_node_script(
        _make_cluster(), resources, "rj-test123", "sft",
    )
    assert "#SBATCH --account=myproject" in script
