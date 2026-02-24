"""Unit tests for server application factory."""

from __future__ import annotations

from server.app import create_app
from server.database import DatabaseConfig


def test_create_app_returns_fastapi_instance() -> None:
    """create_app should return a FastAPI application."""
    config = DatabaseConfig(database_url="sqlite:///:memory:")
    app = create_app(config)
    assert app is not None
    assert hasattr(app, "routes")


def test_create_app_with_default_config() -> None:
    """create_app with None config should use default DatabaseConfig."""
    app = create_app()
    assert app is not None
    assert app.title == "Forge Collaboration Server"
