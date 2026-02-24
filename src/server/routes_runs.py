"""Run CRUD endpoints for Forge collaboration server.

This module defines FastAPI routes for listing, retrieving,
and creating training run records.
"""

from __future__ import annotations

from typing import Any

from core.errors import ForgeDependencyError, ForgeServerError


def _import_fastapi() -> Any:
    """Lazy-import fastapi, raising on absence."""
    try:
        import fastapi
        return fastapi
    except ImportError as exc:
        raise ForgeDependencyError(
            "fastapi is required for the collaboration server. "
            "Install it with: pip install fastapi"
        ) from exc


def create_runs_router(session_factory: Any) -> Any:
    """Create and return a FastAPI router for run endpoints.

    Args:
        session_factory: SQLAlchemy sessionmaker factory.

    Returns:
        Configured FastAPI APIRouter.
    """
    fastapi = _import_fastapi()
    router = fastapi.APIRouter(prefix="/runs", tags=["runs"])

    @router.get("")
    def list_runs() -> list[dict[str, Any]]:
        """List all training runs."""
        from server.models import RunModel
        session = session_factory()
        try:
            runs = session.query(RunModel).all()
            return [
                _serialize_run(run) for run in runs
            ]
        finally:
            session.close()

    @router.get("/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Get a single run by its run_id."""
        from server.models import RunModel
        session = session_factory()
        try:
            run = session.query(RunModel).filter(
                RunModel.run_id == run_id,
            ).first()
            if run is None:
                raise fastapi.HTTPException(
                    status_code=404,
                    detail=f"Run '{run_id}' not found",
                )
            return _serialize_run(run)
        finally:
            session.close()

    @router.post("", status_code=201)
    def create_run(body: dict[str, Any] = fastapi.Body(...)) -> dict[str, Any]:
        """Create a new run record."""
        from server.models import RunModel
        session = session_factory()
        try:
            run = RunModel(
                run_id=body["run_id"],
                dataset_name=body["dataset_name"],
                status=body.get("status", "pending"),
                owner_id=body.get("owner_id"),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return _serialize_run(run)
        except KeyError as exc:
            session.rollback()
            raise fastapi.HTTPException(
                status_code=422,
                detail=f"Missing required field: {exc}",
            )
        except Exception as exc:
            session.rollback()
            raise ForgeServerError(
                f"Failed to create run: {exc}"
            ) from exc
        finally:
            session.close()

    return router


def _serialize_run(run: Any) -> dict[str, Any]:
    """Convert a RunModel to a JSON-safe dictionary.

    Args:
        run: SQLAlchemy RunModel instance.

    Returns:
        Dictionary representation of the run.
    """
    return {
        "id": run.id,
        "run_id": run.run_id,
        "dataset_name": run.dataset_name,
        "status": run.status,
        "created_at": str(run.created_at) if run.created_at else None,
        "updated_at": str(run.updated_at) if run.updated_at else None,
        "owner_id": run.owner_id,
    }
