"""FastAPI application factory for Crucible collaboration server.

This module creates and configures the FastAPI application with
database setup and route registration.
"""

from __future__ import annotations

from typing import Any

from core.errors import CrucibleDependencyError
from server.database import (
    DatabaseConfig,
    create_engine_from_config,
    create_session_factory,
    init_database,
)


def _import_fastapi() -> Any:
    """Lazy-import fastapi, raising on absence."""
    try:
        import fastapi
        return fastapi
    except ImportError as exc:
        raise CrucibleDependencyError(
            "fastapi is required for the collaboration server. "
            "Install it with: pip install fastapi"
        ) from exc


def create_app(config: DatabaseConfig | None = None) -> Any:
    """Create a configured FastAPI application.

    Args:
        config: Optional database configuration. Uses defaults if None.

    Returns:
        Configured FastAPI application instance.

    Raises:
        CrucibleDependencyError: If fastapi is not installed.
    """
    fastapi = _import_fastapi()
    resolved_config = config or DatabaseConfig()

    app = fastapi.FastAPI(
        title="Crucible Collaboration Server",
        description="Real-time collaboration for ML training runs",
        version="0.1.0",
    )

    engine = create_engine_from_config(resolved_config)
    session_factory = create_session_factory(engine)
    init_database(engine)

    _configure_routes(app, session_factory)
    return app


def _configure_routes(app: Any, session_factory: Any) -> None:
    """Include all route modules on the application.

    Args:
        app: FastAPI application instance.
        session_factory: SQLAlchemy sessionmaker factory.
    """
    from server.routes_comments import create_comments_router
    from server.routes_runs import create_runs_router

    app.include_router(create_runs_router(session_factory))
    app.include_router(create_comments_router(session_factory))
