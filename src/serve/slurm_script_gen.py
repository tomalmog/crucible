"""Generate Slurm sbatch scripts for remote job submission.

Supports single-node, multi-node (torchrun), and sweep (job array)
script generation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.remote_env_setup import CONDA_INIT


def generate_single_node_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
    config_filename: str = "training_config.json",
) -> str:
    """Generate an sbatch script for a single-node training job."""
    workdir = f"{cluster.remote_workspace}/{job_id}"
    partition = resources.partition or cluster.default_partition
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    job_name = f"forge-{training_method}-{date_tag}"

    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append(_gpu_gres_line(resources))
    lines.append(f"#SBATCH --cpus-per-task={resources.cpus_per_task}")
    lines.append(f"#SBATCH --mem={resources.memory}")
    lines.append(f"#SBATCH --time={resources.time_limit}")
    lines.append(f"#SBATCH --output={workdir}/slurm-%j.out")
    for key, value in resources.extra_sbatch:
        lines.append(f"#SBATCH --{key}={value}")
    excl = _exclude_line(cluster)
    if excl:
        lines.append(excl)
    lines.append("")
    lines.append("# Merge stderr into stdout so all output is in the log")
    lines.append("exec 2>&1")
    lines.append("")

    lines.extend(_module_load_lines(cluster))
    lines.extend(_cuda_env_lines())
    lines.append(f"cd {workdir}")
    lines.append("tar xzf forge-agent.tar.gz")
    lines.extend(_memory_diagnostic_lines())
    lines.append(f"echo 'FORGE: Starting {training_method} agent...'")
    lines.append(
        f"{cluster.python_path} forge_agent_entry.py --config {config_filename}"
    )
    lines.extend(_post_agent_diagnostic_lines())

    return "\n".join(lines) + "\n"


def generate_multi_node_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
    config_filename: str = "training_config.json",
) -> str:
    """Generate an sbatch script for multi-node distributed training."""
    workdir = f"{cluster.remote_workspace}/{job_id}"
    partition = resources.partition or cluster.default_partition
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    job_name = f"forge-{training_method}-{date_tag}"

    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --nodes={resources.nodes}")
    lines.append("#SBATCH --ntasks-per-node=1")
    lines.append(_gpu_gres_line(resources))
    lines.append(f"#SBATCH --cpus-per-task={resources.cpus_per_task}")
    lines.append(f"#SBATCH --mem={resources.memory}")
    lines.append(f"#SBATCH --time={resources.time_limit}")
    lines.append(f"#SBATCH --output={workdir}/slurm-%j.out")
    for key, value in resources.extra_sbatch:
        lines.append(f"#SBATCH --{key}={value}")
    excl = _exclude_line(cluster)
    if excl:
        lines.append(excl)
    lines.append("")
    lines.append("# Merge stderr into stdout so all output is in the log")
    lines.append("exec 2>&1")
    lines.append("")

    lines.extend(_module_load_lines(cluster))
    lines.extend(_cuda_env_lines())
    lines.append(f"cd {workdir}")
    lines.append("tar xzf forge-agent.tar.gz")
    lines.extend(_memory_diagnostic_lines())
    lines.append(f"echo 'FORGE: Starting {training_method} agent...'")
    lines.append("")

    # Multi-node environment variables
    lines.append("export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)")
    lines.append("export MASTER_PORT=29500")
    lines.append("export NCCL_IB_DISABLE=0")
    lines.append("export NCCL_DEBUG=INFO")
    lines.append("")

    lines.append(
        f"srun {cluster.python_path} -m torch.distributed.run \\"
    )
    lines.append("    --nproc_per_node=$SLURM_GPUS_ON_NODE \\")
    lines.append("    --nnodes=$SLURM_NNODES \\")
    lines.append("    --node_rank=$SLURM_NODEID \\")
    lines.append("    --master_addr=$MASTER_ADDR \\")
    lines.append("    --master_port=$MASTER_PORT \\")
    lines.append(f"    forge_agent_entry.py --config {config_filename}")

    return "\n".join(lines) + "\n"


def generate_sweep_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
    array_size: int,
) -> str:
    """Generate an sbatch script for a sweep using Slurm job arrays.

    Each array task reads its own trial config file based on
    $SLURM_ARRAY_TASK_ID.

    Args:
        cluster: Target cluster configuration.
        resources: Resource allocation per trial.
        job_id: Local job identifier.
        training_method: Training method label.
        array_size: Number of sweep trials.

    Returns:
        Complete sbatch script as a string.
    """
    workdir = f"{cluster.remote_workspace}/{job_id}"
    partition = resources.partition or cluster.default_partition
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    job_name = f"forge-sweep-{training_method}-{date_tag}"

    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --array=0-{array_size - 1}")
    lines.append(_gpu_gres_line(resources))
    lines.append(f"#SBATCH --cpus-per-task={resources.cpus_per_task}")
    lines.append(f"#SBATCH --mem={resources.memory}")
    lines.append(f"#SBATCH --time={resources.time_limit}")
    lines.append(f"#SBATCH --output={workdir}/slurm-%A_%a.out")
    for key, value in resources.extra_sbatch:
        lines.append(f"#SBATCH --{key}={value}")
    excl = _exclude_line(cluster)
    if excl:
        lines.append(excl)
    lines.append("")
    lines.append("# Merge stderr into stdout so all output is in the log")
    lines.append("exec 2>&1")
    lines.append("")

    lines.extend(_module_load_lines(cluster))
    lines.extend(_cuda_env_lines())
    lines.append(f"cd {workdir}")
    lines.append("tar xzf forge-agent.tar.gz")
    lines.extend(_memory_diagnostic_lines())
    lines.append(f"echo 'FORGE: Starting {training_method} agent...'")
    lines.append(
        f"{cluster.python_path} forge_agent_entry.py "
        "--config trials/trial_${SLURM_ARRAY_TASK_ID}.json"
    )
    lines.extend(_post_agent_diagnostic_lines())

    return "\n".join(lines) + "\n"


_CUDA_PREFLIGHT = (
    "python3 << 'CUDA_CHECK' || exit 1\n"
    "import ctypes, sys\n"
    "try:\n"
    "    cuda = ctypes.CDLL('libcuda.so.1')\n"
    "    r = ctypes.c_int()\n"
    "    err = cuda.cuInit(0)\n"
    "    if err != 0:\n"
    "        print('FORGE_AGENT_ERROR: CUDA driver error (cuInit=' + str(err) + '). '\n"
    "              'This node has a broken GPU driver. '\n"
    "              'Resubmit the job to try a different node.', file=sys.stderr)\n"
    "        sys.exit(1)\n"
    "    cuda.cuDeviceGetCount(ctypes.byref(r))\n"
    "    if r.value == 0:\n"
    "        print('FORGE_AGENT_ERROR: No CUDA devices found.', file=sys.stderr)\n"
    "        sys.exit(1)\n"
    "    name = ctypes.create_string_buffer(256)\n"
    "    cuda.cuDeviceGetName(name, 256, 0)\n"
    "    print('CUDA pre-flight OK: ' + str(r.value) + ' device(s), ' + name.value.decode())\n"
    "except Exception as e:\n"
    "    print('FORGE_AGENT_ERROR: CUDA pre-flight failed: ' + str(e), file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "CUDA_CHECK"
)


def _cuda_env_lines() -> list[str]:
    """Set up CUDA environment and run a pre-flight GPU check."""
    return [
        "# Ensure CUDA is available",
        "if ! nvidia-smi > /dev/null 2>&1; then",
        "    module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || true",
        "fi",
        "",
        "export PYTHONUNBUFFERED=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
        "nvidia-smi",
        "",
        "# Pre-flight: verify CUDA driver responds before starting training.",
        "# cuInit returning non-zero means this node has a broken GPU driver.",
        _CUDA_PREFLIGHT,
        "",
    ]


def _memory_diagnostic_lines() -> list[str]:
    """Show available memory before starting the agent."""
    return [
        "echo 'FORGE: Memory available:'",
        "free -m 2>/dev/null | head -3 || true",
        "",
    ]


def _post_agent_diagnostic_lines() -> list[str]:
    """Check exit code and diagnose OOM kills after agent exits."""
    return [
        "rc=$?",
        'if [ $rc -eq 0 ]; then echo "FORGE: Agent finished successfully"',
        "elif [ $rc -eq 137 ] || [ $rc -eq 9 ]; then",
        '    echo "FORGE_AGENT_ERROR: Process killed (exit $rc) — likely OOM killed by Slurm cgroup"',
        '    echo "FORGE: Checking dmesg for OOM events..."',
        "    dmesg -T 2>/dev/null | tail -20 | grep -i 'killed\\|oom\\|memory' || true",
        '    echo "FORGE: Consider increasing --mem in cluster resource settings"',
        'else echo "FORGE: Agent failed with exit code $rc"; fi',
        "exit $rc",
        "",
    ]


def _exclude_line(cluster: ClusterConfig) -> str | None:
    """Return an --exclude SBATCH directive if the cluster has excluded nodes."""
    if cluster.exclude_nodes:
        return f"#SBATCH --exclude={cluster.exclude_nodes}"
    return None


def _gpu_gres_line(resources: SlurmResourceConfig) -> str:
    """Build the --gres=gpu line for sbatch."""
    if resources.gpu_type:
        return (
            f"#SBATCH --gres=gpu:{resources.gpu_type}:{resources.gpus_per_node}"
        )
    return f"#SBATCH --gres=gpu:{resources.gpus_per_node}"


def _module_load_lines(cluster: ClusterConfig) -> list[str]:
    """Build module load and conda activate lines from cluster config."""
    lines: list[str] = []
    for cmd in cluster.module_loads:
        lines.append(cmd)
    lines.append(CONDA_INIT)
    lines.append("conda activate forge")
    lines.append("")
    return lines
