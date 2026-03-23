"""Slurm execution backend — wraps existing remote job infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import CrucibleError
from core.job_types import BackendKind, JobRecord, JobSpec, JobState
from core.slurm_types import SlurmResourceConfig
from store.job_store import generate_job_id, now_iso, save_job, update_job, load_job


def _spec_to_slurm_resources(spec: JobSpec) -> SlurmResourceConfig:
    """Convert JobSpec.resources to SlurmResourceConfig."""
    r = spec.resources
    if r is None:
        return SlurmResourceConfig()
    return SlurmResourceConfig(
        partition=r.partition,
        nodes=r.nodes,
        gpus_per_node=r.gpus_per_node,
        gpu_type=r.gpu_type,
        cpus_per_task=r.cpus_per_task,
        memory=r.memory,
        time_limit=r.time_limit,
        extra_sbatch=r.extra_sbatch,
    )


class SlurmRunner:
    """Delegates to existing remote_job_submitter infrastructure."""

    @property
    def kind(self) -> BackendKind:
        return "slurm"

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord:
        """Submit via existing Slurm infrastructure, write unified JobRecord."""
        resources = _spec_to_slurm_resources(spec)

        if spec.is_sweep and spec.sweep_trials:
            remote_record = self._submit_sweep(
                data_root, spec, resources,
            )
        elif spec.job_type == "eval":
            remote_record = self._submit_eval(
                data_root, spec, resources,
            )
        elif spec.job_type in (
            "logit-lens", "activation-pca", "activation-patch",
        ):
            remote_record = self._submit_interp(
                data_root, spec, resources,
            )
        else:
            remote_record = self._submit_training(
                data_root, spec, resources,
            )

        unified = self._remote_to_unified(remote_record)
        save_job(data_root, unified)
        return unified

    def cancel(self, data_root: Path, job_id: str) -> JobRecord:
        """Cancel via existing infrastructure, update unified record."""
        from serve.remote_model_puller import cancel_remote_job

        record = load_job(data_root, job_id)
        cancel_remote_job(data_root, record.backend_job_id)
        return update_job(data_root, job_id, state="cancelled")

    def get_state(self, data_root: Path, job_id: str) -> JobState:
        """Check state via SSH, update unified record."""
        from serve.remote_job_state import check_remote_job_state

        record = load_job(data_root, job_id)
        # The legacy job_id is stored in backend_job_id field for the
        # remote-jobs store. We need to find the legacy rj- ID.
        legacy_job_id = self._find_legacy_id(data_root, record)
        state = check_remote_job_state(data_root, legacy_job_id)
        update_job(data_root, job_id, state=state)
        return state  # type: ignore[return-value]

    def get_logs(self, data_root: Path, job_id: str, tail: int = 200) -> str:
        """Fetch logs via SSH."""
        from serve.remote_log_streamer import fetch_remote_logs

        record = load_job(data_root, job_id)
        legacy_id = self._find_legacy_id(data_root, record)
        return fetch_remote_logs(data_root, legacy_id, tail_lines=tail)

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]:
        """Fetch result.json via SSH."""
        from serve.remote_result_reader import read_remote_result
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster
        from store.remote_job_store import load_remote_job

        record = load_job(data_root, job_id)
        legacy_id = self._find_legacy_id(data_root, record)
        remote_record = load_remote_job(data_root, legacy_id)
        cluster = load_cluster(data_root, remote_record.cluster_name)
        with SshSession(cluster) as session:
            return read_remote_result(session, remote_record)

    def _find_legacy_id(self, data_root: Path, record: JobRecord) -> str:
        """Find the legacy rj- job ID from backend_job_id or job_id."""
        # If this record was migrated, backend_job_id holds the legacy ID.
        # If submitted through new path, we stored it there too.
        if record.backend_job_id.startswith("rj-"):
            return record.backend_job_id
        # Fall back: check if the job_id itself is a legacy ID
        if record.job_id.startswith("rj-"):
            return record.job_id
        raise CrucibleError(
            f"Cannot find legacy remote job ID for {record.job_id}"
        )

    def _submit_training(
        self, data_root: Path, spec: JobSpec,
        resources: SlurmResourceConfig,
    ) -> object:
        from serve.remote_job_submitter import submit_remote_job

        return submit_remote_job(
            data_root=data_root,
            cluster_name=spec.cluster_name,
            training_method=spec.job_type,
            method_args=dict(spec.method_args),
            resources=resources,
            model_name=spec.label,
        )

    def _submit_eval(
        self, data_root: Path, spec: JobSpec,
        resources: SlurmResourceConfig,
    ) -> object:
        from serve.remote_job_submitter import submit_remote_eval_job

        return submit_remote_eval_job(
            data_root=data_root,
            cluster_name=spec.cluster_name,
            method_args=dict(spec.method_args),
            resources=resources,
            model_name=spec.label,
        )

    def _submit_interp(
        self, data_root: Path, spec: JobSpec,
        resources: SlurmResourceConfig,
    ) -> object:
        from serve.remote_job_submitter import submit_remote_interp_job

        return submit_remote_interp_job(
            data_root=data_root,
            cluster_name=spec.cluster_name,
            interp_method=spec.job_type,
            method_args=dict(spec.method_args),
            resources=resources,
        )

    def _submit_sweep(
        self, data_root: Path, spec: JobSpec,
        resources: SlurmResourceConfig,
    ) -> object:
        from serve.remote_job_submitter import submit_remote_sweep

        return submit_remote_sweep(
            data_root=data_root,
            cluster_name=spec.cluster_name,
            training_method=spec.job_type,
            trial_configs=list(spec.sweep_trials),
            resources=resources,
        )

    def _remote_to_unified(self, remote_record: object) -> JobRecord:
        """Convert a RemoteJobRecord to a unified JobRecord."""
        from core.slurm_types import RemoteJobRecord
        if not isinstance(remote_record, RemoteJobRecord):
            raise CrucibleError("Expected RemoteJobRecord from Slurm submit")
        ts = now_iso()
        return JobRecord(
            job_id=generate_job_id(),
            backend="slurm",
            job_type=remote_record.training_method,
            state=remote_record.state,  # type: ignore[arg-type]
            created_at=remote_record.submitted_at,
            updated_at=ts,
            label=remote_record.model_name,
            backend_job_id=remote_record.job_id,
            backend_cluster=remote_record.cluster_name,
            backend_output_dir=remote_record.remote_output_dir,
            backend_log_path=remote_record.remote_log_path,
            model_path=remote_record.model_path_remote,
            model_path_local=remote_record.model_path_local,
            model_name=remote_record.model_name,
            submit_phase=remote_record.submit_phase,
            is_sweep=remote_record.is_sweep,
            sweep_trial_count=remote_record.sweep_array_size,
        )
