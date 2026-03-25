"""SSH execution backend — runs jobs directly or inside Docker over SSH."""

from __future__ import annotations

import json
from pathlib import Path

from core.constants import DEFAULT_DOCKER_IMAGE, sanitize_remote_name
from core.errors import CrucibleDockerError, CrucibleRemoteError
from core.job_types import BackendKind, JobRecord, JobSpec, JobState
from core.slurm_types import ClusterConfig
from store.job_store import generate_job_id, now_iso, save_job, load_job, update_job


def _uses_docker(cluster: ClusterConfig) -> bool:
    """Return True when the cluster is configured to run inside Docker."""
    return bool(cluster.docker_image)


class SshRunner:
    """Runs jobs on remote hosts via SSH, optionally inside Docker."""

    @property
    def kind(self) -> BackendKind:
        return "ssh"

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord:
        """Submit a job over SSH (training, eval, interp, or sweep)."""
        from store.cluster_registry import load_cluster

        cluster = load_cluster(data_root, spec.cluster_name)
        if _uses_docker(cluster):
            return self._submit_docker(data_root, spec, cluster)
        return self._submit_bare(data_root, spec, cluster)

    def _submit_docker(
        self, data_root: Path, spec: JobSpec, cluster: ClusterConfig,
    ) -> JobRecord:
        """Submit via Docker container on the remote host."""
        from serve.agent_bundler import build_agent_tarball
        from serve.docker_commands import (
            build_docker_run_cmd, build_gpu_flags, parse_container_id,
        )
        from serve.ssh_connection import SshSession
        from serve.ssh_submit_helpers import build_docker_container_cmd

        image = cluster.docker_image or DEFAULT_DOCKER_IMAGE
        job_id, job_dir, is_sweep = self._init_job(spec, cluster)
        tarball_path = build_agent_tarball(data_root / "cache" / "agent")
        with SshSession(cluster) as session:
            resolved_dir = session.resolve_path(job_dir)
            session.mkdir_p(resolved_dir)
            session.mkdir_p(f"{resolved_dir}/output")
            session.upload(
                tarball_path, f"{resolved_dir}/crucible-agent.tar.gz",
            )

            container_cmd = build_docker_container_cmd(
                session, resolved_dir, spec, is_sweep,
                python_path=cluster.python_path,
            )
            cmd = build_docker_run_cmd(
                image=image,
                gpu_flags=build_gpu_flags(""),
                volumes=(
                    (resolved_dir, "/workspace"),
                    (f"{resolved_dir}/output", "/output"),
                ),
                workdir="/workspace",
                command=container_cmd,
            )
            stdout, stderr, exit_code = session.execute(cmd, timeout=30)
            if exit_code != 0:
                raise CrucibleDockerError(
                    f"Docker run failed on {cluster.host}: {stderr.strip()}"
                )
            backend_job_id = parse_container_id(stdout)

        return self._finalize_docker_record(
            data_root, job_id, spec, job_dir, backend_job_id, is_sweep,
        )

    def _submit_bare(
        self, data_root: Path, spec: JobSpec, cluster: ClusterConfig,
    ) -> JobRecord:
        """Submit via bare SSH using the Crucible agent tarball."""
        from serve.ssh_submit_helpers import run_bare_submission

        job_id, job_dir, is_sweep = self._init_job(spec, cluster)

        # Save record immediately so the UI shows the job during setup
        self._save_early_record(
            data_root, job_id, spec, job_dir, is_sweep,
        )

        try:
            run_bare_submission(
                data_root, job_id, spec, cluster, job_dir, is_sweep,
            )
        except Exception as exc:
            update_job(
                data_root, job_id,
                state="failed",
                error_message=f"{type(exc).__name__}: {exc}",
                submit_phase="",
            )
            raise

        return load_job(data_root, job_id)

    def cancel(self, data_root: Path, job_id: str) -> JobRecord:
        """Stop the remote job and mark it cancelled."""
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        with SshSession(cluster) as session:
            if _uses_docker(cluster):
                cmd = f"docker stop {record.backend_job_id}"
                _, stderr, exit_code = session.execute(cmd, timeout=30)
                if exit_code != 0:
                    raise CrucibleDockerError(
                        f"Docker stop failed: {stderr.strip()}"
                    )
            else:
                session.execute(
                    f"kill {record.backend_job_id} 2>/dev/null; true",
                    timeout=15,
                )

        return update_job(data_root, job_id, state="cancelled")

    def get_state(self, data_root: Path, job_id: str) -> JobState:
        """Query the remote job state via SSH."""
        from core.errors import CrucibleRemoteError
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)

        # Job is still being submitted — don't probe the remote yet
        if record.state == "submitting":
            return "submitting"

        cluster = load_cluster(data_root, record.backend_cluster)

        try:
            with SshSession(cluster) as session:
                if _uses_docker(cluster):
                    state = _get_docker_state(session, record)
                else:
                    state = _get_bare_state(session, record)

                if state == "completed" and not record.model_path:
                    _handle_completion(session, data_root, record)
        except (CrucibleRemoteError, OSError) as exc:
            # SSH connection failed — return current state so we don't
            # falsely mark the job as failed due to a network hiccup.
            import logging
            logging.getLogger(__name__).warning(
                "SSH sync failed for job %s: %s", job_id, exc,
            )
            return record.state

        update_job(data_root, job_id, state=state)
        return state

    def get_logs(
        self, data_root: Path, job_id: str, tail: int = 200,
    ) -> str:
        """Fetch job logs via SSH."""
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        with SshSession(cluster) as session:
            if _uses_docker(cluster):
                cmd = f"docker logs {record.backend_job_id} --tail {tail}"
                stdout, stderr, _ = session.execute(cmd, timeout=30)
                return stdout or stderr

            log_path = session.resolve_path(
                f"{record.backend_output_dir}/output/train.log",
            )
            stdout, _, _ = session.execute(
                f"tail -n {tail} {log_path}", timeout=30,
            )
            return stdout

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]:
        """Read result.json from the job output directory."""
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        with SshSession(cluster) as session:
            result_path = session.resolve_path(
                f"{record.backend_output_dir}/output/result.json",
            )
            stdout, stderr, exit_code = session.execute(
                f"cat {result_path}", timeout=15,
            )
            if exit_code != 0:
                # No result.json — try to get context from logs
                log_path = session.resolve_path(
                    f"{record.backend_output_dir}/output/train.log",
                )
                log_out, _, _ = session.execute(
                    f"tail -30 {log_path} 2>/dev/null", timeout=10,
                )
                return {
                    "status": "failed",
                    "error": "No result.json produced by the job",
                    "logs_tail": log_out.strip() if log_out else "",
                }

        return json.loads(stdout)  # type: ignore[no-any-return]

    @staticmethod
    def _init_job(
        spec: JobSpec, cluster: ClusterConfig,
    ) -> tuple[str, str, bool]:
        """Generate job ID, directory, and sweep flag."""
        workspace = cluster.remote_workspace or "~/crucible-jobs"
        job_id = generate_job_id()
        job_dir = f"{workspace}/{sanitize_remote_name(job_id)}"
        is_sweep = spec.is_sweep and len(spec.sweep_trials) > 0
        return job_id, job_dir, is_sweep

    def _save_early_record(
        self, data_root: Path, job_id: str, spec: JobSpec,
        job_dir: str, is_sweep: bool,
    ) -> JobRecord:
        """Save an initial 'submitting' record so the UI shows it immediately."""
        ts = now_iso()
        record = JobRecord(
            job_id=job_id,
            backend="ssh",
            job_type=spec.job_type,
            state="submitting",
            created_at=ts,
            updated_at=ts,
            label=spec.label or spec.job_type,
            model_name=spec.label or "",
            backend_cluster=spec.cluster_name,
            backend_output_dir=job_dir,
            is_sweep=is_sweep,
            sweep_trial_count=len(spec.sweep_trials) if is_sweep else 0,
            config=spec.config or {},
            submit_phase="Connecting to cluster...",
        )
        save_job(data_root, record)
        return record

    def _finalize_docker_record(
        self, data_root: Path, job_id: str, spec: JobSpec,
        job_dir: str, backend_job_id: str, is_sweep: bool,
    ) -> JobRecord:
        """Persist a job record for Docker submissions."""
        ts = now_iso()
        record = JobRecord(
            job_id=job_id,
            backend="ssh",
            job_type=spec.job_type,
            state="running",
            created_at=ts,
            updated_at=ts,
            label=spec.label or spec.job_type,
            model_name=spec.label or "",
            backend_job_id=backend_job_id,
            backend_cluster=spec.cluster_name,
            backend_output_dir=job_dir,
            is_sweep=is_sweep,
            sweep_trial_count=len(spec.sweep_trials) if is_sweep else 0,
            config=spec.config or {},
        )
        save_job(data_root, record)
        return record


def _get_docker_state(session: object, record: JobRecord) -> JobState:
    from serve.ssh_state_helpers import get_docker_state
    return get_docker_state(session, record)


def _get_bare_state(session: object, record: JobRecord) -> JobState:
    from serve.ssh_state_helpers import get_bare_state
    return get_bare_state(session, record)


def _handle_completion(
    session: object, data_root: Path, record: JobRecord,
) -> None:
    from serve.ssh_state_helpers import handle_ssh_completion
    handle_ssh_completion(session, data_root, record)


