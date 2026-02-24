"""Unit tests for server ORM models."""

from __future__ import annotations

from server.models import (
    AnnotationModel,
    Base,
    CommentModel,
    RunModel,
    UserModel,
    WorkspaceModel,
)


def test_model_classes_exist() -> None:
    """All expected ORM model classes should be importable."""
    assert UserModel is not None
    assert RunModel is not None
    assert CommentModel is not None
    assert AnnotationModel is not None
    assert WorkspaceModel is not None


def test_user_model_has_expected_columns() -> None:
    """UserModel should define id, username, api_key, created_at."""
    columns = {c.name for c in UserModel.__table__.columns}
    assert "id" in columns
    assert "username" in columns
    assert "api_key" in columns
    assert "created_at" in columns


def test_run_model_has_expected_columns() -> None:
    """RunModel should define run_id, dataset_name, status, owner_id."""
    columns = {c.name for c in RunModel.__table__.columns}
    assert "id" in columns
    assert "run_id" in columns
    assert "dataset_name" in columns
    assert "status" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "owner_id" in columns


def test_base_has_all_tables() -> None:
    """Base metadata should contain all expected table names."""
    table_names = set(Base.metadata.tables.keys())
    assert "users" in table_names
    assert "runs" in table_names
    assert "comments" in table_names
    assert "annotations" in table_names
    assert "workspaces" in table_names
