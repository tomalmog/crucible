"""Data transfer helpers for remote Slurm job submission.

Handles uploading agent bundles, configs, and scripts to remote clusters.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.slurm_script_gen import (
    generate_multi_node_script,
    generate_single_node_script,
)
from serve.ssh_connection import SshSession


def _upload_bundle(session: SshSession, tarball: Path, workdir: str) -> None:
    """Upload the agent tarball to the remote workspace."""
    session.upload(tarball, f"{workdir}/crucible-agent.tar.gz")


def _upload_config(
    session: SshSession,
    config: dict[str, object],
    workdir: str,
    filename: str = "training_config.json",
) -> None:
    """Write and upload a training config JSON file."""
    session.upload_text(json.dumps(config, indent=2), f"{workdir}/{filename}")


def _upload_script(session: SshSession, script: str, workdir: str) -> None:
    """Write and upload an sbatch script."""
    session.upload_text(script, f"{workdir}/job.sh")


def _submit_sbatch(session: SshSession, workdir: str) -> str:
    """Submit the sbatch script and parse the Slurm job ID."""
    stdout, stderr, code = session.execute(
        f"sbatch {shlex.quote(workdir + '/job.sh')}", timeout=30,
    )
    if code != 0:
        raise CrucibleRemoteError(f"sbatch failed: {stderr.strip()}")
    # sbatch output: "Submitted batch job 12345"
    parts = stdout.strip().split()
    if len(parts) < 4:
        raise CrucibleRemoteError(f"Unexpected sbatch output: {stdout.strip()}")
    return parts[-1]


def _generate_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
) -> str:
    """Select and generate the appropriate sbatch script."""
    if resources.nodes > 1:
        return generate_multi_node_script(
            cluster, resources, job_id, training_method,
        )
    return generate_single_node_script(
        cluster, resources, job_id, training_method,
    )


