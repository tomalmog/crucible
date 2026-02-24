"""Comment and annotation endpoints for Forge collaboration server.

This module defines FastAPI routes for listing and creating
comments and annotations on training runs.
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


def create_comments_router(session_factory: Any) -> Any:
    """Create and return a FastAPI router for comment endpoints.

    Args:
        session_factory: SQLAlchemy sessionmaker factory.

    Returns:
        Configured FastAPI APIRouter.
    """
    fastapi = _import_fastapi()
    router = fastapi.APIRouter(tags=["comments"])

    @router.get("/runs/{run_id}/comments")
    def list_comments(run_id: str) -> list[dict[str, Any]]:
        """List all comments for a run."""
        from server.models import CommentModel, RunModel
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
            comments = session.query(CommentModel).filter(
                CommentModel.run_id == run.id,
            ).all()
            return [_serialize_comment(c) for c in comments]
        finally:
            session.close()

    @router.post("/runs/{run_id}/comments", status_code=201)
    def add_comment(
        run_id: str,
        body: dict[str, Any] = fastapi.Body(...),
    ) -> dict[str, Any]:
        """Add a comment to a run."""
        from server.models import CommentModel, RunModel
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
            comment = CommentModel(
                run_id=run.id,
                user_id=body["user_id"],
                content=body["content"],
            )
            session.add(comment)
            session.commit()
            session.refresh(comment)
            return _serialize_comment(comment)
        except KeyError as exc:
            session.rollback()
            raise fastapi.HTTPException(
                status_code=422,
                detail=f"Missing required field: {exc}",
            )
        except fastapi.HTTPException:
            raise
        except Exception as exc:
            session.rollback()
            raise ForgeServerError(
                f"Failed to add comment: {exc}"
            ) from exc
        finally:
            session.close()

    @router.get("/runs/{run_id}/annotations")
    def list_annotations(run_id: str) -> list[dict[str, Any]]:
        """List all annotations for a run."""
        from server.models import AnnotationModel, RunModel
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
            annotations = session.query(AnnotationModel).filter(
                AnnotationModel.run_id == run.id,
            ).all()
            return [_serialize_annotation(a) for a in annotations]
        finally:
            session.close()

    @router.post("/runs/{run_id}/annotations", status_code=201)
    def add_annotation(
        run_id: str,
        body: dict[str, Any] = fastapi.Body(...),
    ) -> dict[str, Any]:
        """Add an annotation to a run."""
        from server.models import AnnotationModel, RunModel
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
            annotation = AnnotationModel(
                run_id=run.id,
                user_id=body["user_id"],
                key=body["key"],
                value=body["value"],
            )
            session.add(annotation)
            session.commit()
            session.refresh(annotation)
            return _serialize_annotation(annotation)
        except KeyError as exc:
            session.rollback()
            raise fastapi.HTTPException(
                status_code=422,
                detail=f"Missing required field: {exc}",
            )
        except fastapi.HTTPException:
            raise
        except Exception as exc:
            session.rollback()
            raise ForgeServerError(
                f"Failed to add annotation: {exc}"
            ) from exc
        finally:
            session.close()

    return router


def _serialize_comment(comment: Any) -> dict[str, Any]:
    """Convert a CommentModel to a JSON-safe dictionary."""
    return {
        "id": comment.id,
        "run_id": comment.run_id,
        "user_id": comment.user_id,
        "content": comment.content,
        "created_at": str(comment.created_at) if comment.created_at else None,
    }


def _serialize_annotation(annotation: Any) -> dict[str, Any]:
    """Convert an AnnotationModel to a JSON-safe dictionary."""
    return {
        "id": annotation.id,
        "run_id": annotation.run_id,
        "user_id": annotation.user_id,
        "key": annotation.key,
        "value": annotation.value,
        "created_at": (
            str(annotation.created_at) if annotation.created_at else None
        ),
    }
