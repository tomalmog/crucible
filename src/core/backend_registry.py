"""Registry for execution backends. One import, one lookup."""

from __future__ import annotations

from core.errors import CrucibleError
from core.job_types import BackendKind
from core.runner_protocol import ExecutionBackend


_BACKENDS: dict[BackendKind, ExecutionBackend] = {}


def register_backend(kind: BackendKind, backend: ExecutionBackend) -> None:
    """Register an execution backend instance by kind."""
    _BACKENDS[kind] = backend


def get_backend(kind: BackendKind) -> ExecutionBackend:
    """Look up a registered backend. Raises CrucibleError if not found."""
    try:
        return _BACKENDS[kind]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise CrucibleError(
            f"Unknown backend '{kind}'. Available: {available}"
        )
