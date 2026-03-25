"""Submission helpers for the SSH execution backend.

Builds commands, uploads configs, and prepares sweep scripts for both
bare SSH and Docker submission paths.  Used exclusively by SshRunner.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import CrucibleRemoteError
from core.job_types import JobSpec
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession


def provision_env(session: SshSession, cluster: ClusterConfig) -> str:
    """Provision the Python environment and return a shell activation prefix.

    Strategy:
    1. Try existing conda → create/use crucible env (HPC clusters).
    2. If conda not installed, install Miniconda first, then create env.
    """
    from serve.remote_env_setup import CONDA_ACTIVATE, ensure_remote_env

    # Try existing conda
    try:
        ensure_remote_env(session)
        return CONDA_ACTIVATE
    except CrucibleRemoteError:
        pass

    # Conda not available — install Miniconda, then retry
    print("SSH_RUNNER: conda not found — installing Miniconda...", flush=True)
    _install_miniconda(session)
    ensure_remote_env(session)
    return CONDA_ACTIVATE


def _install_miniconda(session: SshSession) -> None:
    """Download and install Miniconda on the remote host."""
    # Detect architecture
    stdout, _, _ = session.execute("uname -m", timeout=10)
    arch = stdout.strip()
    if arch == "aarch64":
        installer = "Miniconda3-latest-Linux-aarch64.sh"
    else:
        installer = "Miniconda3-latest-Linux-x86_64.sh"

    url = f"https://repo.anaconda.com/miniconda/{installer}"
    print(f"SSH_RUNNER: Downloading {installer}...", flush=True)
    _, stderr, code = session.execute(
        f"curl -fsSL -o /tmp/miniconda.sh {url}", timeout=120,
    )
    if code != 0:  # fallback to wget
        _, stderr, code = session.execute(
            f"wget -q -O /tmp/miniconda.sh {url}", timeout=120,
        )
    if code != 0:
        raise CrucibleRemoteError(f"Failed to download Miniconda: {stderr.strip()}")

    print("SSH_RUNNER: Installing Miniconda...", flush=True)
    _, stderr, code = session.execute(
        "bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh",
        timeout=300,
    )
    if code != 0:
        raise CrucibleRemoteError(f"Miniconda install failed: {stderr.strip()}")
    print("SSH_RUNNER: Miniconda installed.", flush=True)


def prepare_single_config(
    session: SshSession,
    resolved_dir: str,
    spec: JobSpec,
    python_path: str = "python3",
    env_activate: str = "",
) -> str:
    """Upload a single job config and return the nohup launch command."""
    config = {
        "method": spec.job_type,
        "method_args": dict(spec.method_args),
        "result_output": "output/result.json",
    }
    session.upload_text(
        json.dumps(config, indent=2),
        f"{resolved_dir}/training_config.json",
    )
    activate = f"{env_activate} && " if env_activate else ""
    # Use a wrapper script so the SSH channel closes immediately while the
    # agent runs in the background.  Writing stdout/stderr/stdin redirection
    # and nohup inside a bash -c wrapper with exec ensures no inherited FDs
    # keep the SSH channel alive.
    wrapper = (
        f"#!/bin/bash\n"
        f"{activate.rstrip(' &').rstrip()} \n"
        f"nohup {python_path} crucible_agent_entry.py "
        f"--config training_config.json "
        f"> output/train.log 2>&1 &\n"
        f"echo $! > .pid\n"
    )
    session.upload_text(wrapper, f"{resolved_dir}/launch.sh")
    return (
        f"cd {resolved_dir} && "
        f"tar xzf crucible-agent.tar.gz && "
        f"bash launch.sh && cat .pid"
    )


def prepare_sweep_configs(
    session: SshSession,
    resolved_dir: str,
    spec: JobSpec,
    python_path: str = "python3",
    env_activate: str = "",
) -> str:
    """Upload per-trial configs and return a sweep launch command.

    Runs each trial sequentially in a single background process.
    Each trial writes to output/trial_N/result.json, and the
    wrapper writes a combined output/result.json on completion.
    """
    session.mkdir_p(f"{resolved_dir}/trials")
    for idx, trial_args in enumerate(spec.sweep_trials):
        merged = {**dict(spec.method_args), **dict(trial_args)}
        config = {
            "method": spec.job_type,
            "method_args": merged,
            "result_output": f"output/trial_{idx}/result.json",
        }
        session.upload_text(
            json.dumps(config, indent=2),
            f"{resolved_dir}/trials/trial_{idx}.json",
        )
        session.mkdir_p(f"{resolved_dir}/output/trial_{idx}")

    num_trials = len(spec.sweep_trials)
    sweep_script = build_sweep_script(
        resolved_dir, num_trials, python_path, env_activate,
    )
    session.upload_text(sweep_script, f"{resolved_dir}/run_sweep.sh")

    activate = f"{env_activate} && " if env_activate else ""
    wrapper = (
        f"#!/bin/bash\n"
        f"{activate.rstrip(' &').rstrip()} \n"
        f"nohup bash run_sweep.sh "
        f"> output/train.log 2>&1 &\n"
        f"echo $! > .pid\n"
    )
    session.upload_text(wrapper, f"{resolved_dir}/launch.sh")
    return (
        f"cd {resolved_dir} && "
        f"tar xzf crucible-agent.tar.gz && "
        f"bash launch.sh && cat .pid"
    )


def build_docker_container_cmd(
    session: SshSession,
    resolved_dir: str,
    spec: JobSpec,
    is_sweep: bool,
    python_path: str = "python3",
) -> str:
    """Build the command to run inside a Docker container.

    For sweeps, uploads per-trial configs and a sweep runner script.
    For single jobs, uploads a training config.

    Returns:
        Shell command string to pass to ``docker run``.
    """
    if is_sweep:
        prepare_sweep_configs(session, resolved_dir, spec, python_path)
        num_trials = len(spec.sweep_trials)
        sweep_script = build_sweep_script("/workspace", num_trials, python_path)
        session.upload_text(sweep_script, f"{resolved_dir}/run_sweep.sh")
        return (
            "cd /workspace && tar xzf crucible-agent.tar.gz "
            "&& bash run_sweep.sh 2>&1 | tee /output/train.log"
        )

    config = {
        "method": spec.job_type,
        "method_args": dict(spec.method_args),
        "result_output": "output/result.json",
    }
    session.upload_text(
        json.dumps(config, indent=2),
        f"{resolved_dir}/training_config.json",
    )
    return (
        "cd /workspace && tar xzf crucible-agent.tar.gz "
        f"&& {python_path} crucible_agent_entry.py "
        "--config training_config.json "
        "2>&1 | tee /output/train.log"
    )


def build_sweep_script(
    resolved_dir: str,
    num_trials: int,
    python_path: str = "python3",
    env_activate: str = "",
) -> str:
    """Generate a bash script that runs sweep trials sequentially.

    Each trial runs the agent entry point with its own config and writes
    results to output/trial_N/.  A final combined result.json is written
    to output/result.json so get_result() works uniformly.
    """
    lines = [
        "#!/usr/bin/env bash",
        f'cd "{resolved_dir}"',
    ]
    if env_activate:
        lines.append(env_activate)
    lines.extend([
        "failed=0",
        "completed=0",
    ])
    for idx in range(num_trials):
        lines.extend([
            f'echo "CRUCIBLE_AGENT: Starting trial {idx}/{num_trials}"',
            f"mkdir -p output/trial_{idx}",
            (
                f"if {python_path} crucible_agent_entry.py "
                f"--config trials/trial_{idx}.json; then"
            ),
            "  completed=$((completed+1))",
            "else",
            "  failed=$((failed+1))",
            "fi",
        ])
    # Write combined result — use printf to embed shell variables in JSON
    lines.extend([
        (
            "printf "
            "'{\"status\": \"completed\", "
            f"\"total_trials\": {num_trials}, "
            "\"completed\": %d, "
            "\"failed\": %d}\\n' "
            "\"$completed\" \"$failed\" "
            "> output/result.json"
        ),
        'echo "CRUCIBLE_AGENT_COMPLETE"',
    ])
    return "\n".join(lines) + "\n"


def run_bare_submission(
    data_root: Path,
    job_id: str,
    spec: JobSpec,
    cluster: ClusterConfig,
    job_dir: str,
    is_sweep: bool,
) -> None:
    """Execute the SSH submission steps, updating submit_phase as we go.

    Called by SshRunner._submit_bare after the early record is saved.
    """
    from serve.agent_bundler import build_agent_tarball
    from store.job_store import update_job

    def phase(msg: str) -> None:
        update_job(data_root, job_id, submit_phase=msg)

    phase("Provisioning environment...")
    tarball_path = build_agent_tarball(data_root / "cache" / "agent")

    with SshSession(cluster) as session:
        env_prefix = provision_env(session, cluster)

        # Resolve dataset: push if needed, set raw_data_path + method data path
        from serve.ssh_dataset_resolver import resolve_dataset
        resolve_dataset(session, cluster, data_root, spec, phase)

        # Resolve ~ in base_model_path so the agent uses an absolute path
        bmp = str(spec.method_args.get("base_model_path", ""))
        if bmp:
            spec.method_args["base_model_path"] = session.resolve_path(bmp)

        phase("Uploading agent...")
        resolved_dir = session.resolve_path(job_dir)
        session.mkdir_p(resolved_dir)
        session.mkdir_p(f"{resolved_dir}/output")
        session.upload(
            tarball_path, f"{resolved_dir}/crucible-agent.tar.gz",
        )

        phase("Launching training...")
        py = cluster.python_path
        if is_sweep:
            run_cmd = prepare_sweep_configs(
                session, resolved_dir, spec,
                python_path=py, env_activate=env_prefix,
            )
        else:
            run_cmd = prepare_single_config(
                session, resolved_dir, spec,
                python_path=py, env_activate=env_prefix,
            )

        stdout, stderr, exit_code = session.execute(run_cmd, timeout=120)
        if exit_code != 0:
            raise CrucibleRemoteError(
                f"Agent launch failed on {cluster.host}: {stderr.strip()}"
            )
        pid = stdout.strip().splitlines()[-1].strip()

    update_job(
        data_root, job_id,
        state="running",
        backend_job_id=pid,
        submit_phase="",
    )
