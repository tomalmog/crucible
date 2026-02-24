"""SQLAlchemy ORM models for Forge collaboration server.

This module defines the relational schema for users, runs,
comments, annotations, and workspaces using declarative mapping.
"""

from __future__ import annotations

from typing import Any

from core.errors import ForgeDependencyError


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


def _build_base() -> Any:
    """Create and return the SQLAlchemy declarative base."""
    _import_sqlalchemy()
    from sqlalchemy.orm import declarative_base
    return declarative_base()


Base = _build_base()

_sa = _import_sqlalchemy()


class UserModel(Base):  # type: ignore[misc]
    """Registered user with API key for authentication.

    Attributes:
        id: Primary key.
        username: Unique username.
        api_key: Secret key for API authentication.
        created_at: Account creation timestamp.
    """

    __tablename__ = "users"

    id = _sa.Column(_sa.Integer, primary_key=True, autoincrement=True)
    username = _sa.Column(_sa.String(255), unique=True, nullable=False)
    api_key = _sa.Column(_sa.String(255), unique=True, nullable=False)
    created_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(), nullable=False,
    )


class RunModel(Base):  # type: ignore[misc]
    """Training run record for collaboration tracking.

    Attributes:
        id: Primary key.
        run_id: External run identifier.
        dataset_name: Associated dataset name.
        status: Current run status.
        created_at: Run creation timestamp.
        updated_at: Last status update timestamp.
        owner_id: Foreign key to owning user.
    """

    __tablename__ = "runs"

    id = _sa.Column(_sa.Integer, primary_key=True, autoincrement=True)
    run_id = _sa.Column(_sa.String(255), unique=True, nullable=False)
    dataset_name = _sa.Column(_sa.String(255), nullable=False)
    status = _sa.Column(
        _sa.String(50), nullable=False, default="pending",
    )
    created_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(), nullable=False,
    )
    updated_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(),
        onupdate=_sa.func.now(), nullable=False,
    )
    owner_id = _sa.Column(
        _sa.Integer, _sa.ForeignKey("users.id"), nullable=True,
    )


class CommentModel(Base):  # type: ignore[misc]
    """Comment attached to a training run.

    Attributes:
        id: Primary key.
        run_id: Foreign key to the run.
        user_id: Foreign key to the commenting user.
        content: Comment text body.
        created_at: Comment creation timestamp.
    """

    __tablename__ = "comments"

    id = _sa.Column(_sa.Integer, primary_key=True, autoincrement=True)
    run_id = _sa.Column(
        _sa.Integer, _sa.ForeignKey("runs.id"), nullable=False,
    )
    user_id = _sa.Column(
        _sa.Integer, _sa.ForeignKey("users.id"), nullable=False,
    )
    content = _sa.Column(_sa.Text, nullable=False)
    created_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(), nullable=False,
    )


class AnnotationModel(Base):  # type: ignore[misc]
    """Key-value annotation on a training run.

    Attributes:
        id: Primary key.
        run_id: Foreign key to the run.
        user_id: Foreign key to the annotating user.
        key: Annotation key.
        value: Annotation value.
        created_at: Annotation creation timestamp.
    """

    __tablename__ = "annotations"

    id = _sa.Column(_sa.Integer, primary_key=True, autoincrement=True)
    run_id = _sa.Column(
        _sa.Integer, _sa.ForeignKey("runs.id"), nullable=False,
    )
    user_id = _sa.Column(
        _sa.Integer, _sa.ForeignKey("users.id"), nullable=False,
    )
    key = _sa.Column(_sa.String(255), nullable=False)
    value = _sa.Column(_sa.Text, nullable=False)
    created_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(), nullable=False,
    )


class WorkspaceModel(Base):  # type: ignore[misc]
    """Named workspace for organizing runs.

    Attributes:
        id: Primary key.
        name: Workspace display name.
        created_at: Workspace creation timestamp.
    """

    __tablename__ = "workspaces"

    id = _sa.Column(_sa.Integer, primary_key=True, autoincrement=True)
    name = _sa.Column(_sa.String(255), unique=True, nullable=False)
    created_at = _sa.Column(
        _sa.DateTime, server_default=_sa.func.now(), nullable=False,
    )
