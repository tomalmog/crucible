"""Protocol definition for pluggable execution backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from core.job_types import BackendKind, JobRecord, JobSpec, JobState


class ExecutionBackend(Protocol):
    """Interface every execution backend must implement."""

    @property
    def kind(self) -> BackendKind: ...

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord: ...

    def cancel(self, data_root: Path, job_id: str) -> JobRecord: ...

    def get_state(self, data_root: Path, job_id: str) -> JobState: ...

    def get_logs(
        self, data_root: Path, job_id: str, tail: int = 200,
    ) -> str: ...

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]: ...
