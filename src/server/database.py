"""Database engine and session factory for Forge collaboration server.

This module provides SQLAlchemy engine creation, session management,
and schema initialization with lazy imports for optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import ForgeDependencyError, ForgeServerError


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration.

    Attributes:
        database_url: SQLAlchemy connection string.
    """

    database_url: str = "sqlite:///forge_server.db"


def _import_sqlalchemy() -> Any:
    """Lazy-import sqlalchemy, raising on absence."""
    try:
        import sqlalchemy
        return sqlalchemy
    except ImportError as exc:
        raise ForgeDependencyError(
            "sqlalchemy is required for the collaboration server. "
            "Install it with: pip install sqlalchemy"
        ) from exc


def create_engine_from_config(config: DatabaseConfig) -> Any:
    """Create a SQLAlchemy engine from configuration.

    Args:
        config: Database configuration with connection URL.

    Returns:
        SQLAlchemy engine instance.

    Raises:
        ForgeDependencyError: If sqlalchemy is not installed.
        ForgeServerError: If engine creation fails.
    """
    sa = _import_sqlalchemy()
    try:
        return sa.create_engine(config.database_url)
    except Exception as exc:
        raise ForgeServerError(
            f"Failed to create database engine: {exc}"
        ) from exc


def create_session_factory(engine: Any) -> Any:
    """Create a SQLAlchemy sessionmaker bound to the engine.

    Args:
        engine: SQLAlchemy engine instance.

    Returns:
        Configured sessionmaker factory.

    Raises:
        ForgeDependencyError: If sqlalchemy is not installed.
    """
    sa = _import_sqlalchemy()
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(bind=engine)


def init_database(engine: Any) -> None:
    """Create all tables defined by ORM models.

    Args:
        engine: SQLAlchemy engine instance.

    Raises:
        ForgeDependencyError: If sqlalchemy is not installed.
        ForgeServerError: If table creation fails.
    """
    _import_sqlalchemy()
    from server.models import Base
    try:
        Base.metadata.create_all(engine)
    except Exception as exc:
        raise ForgeServerError(
            f"Failed to initialize database schema: {exc}"
        ) from exc
