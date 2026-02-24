"""API key authentication for Forge collaboration server.

This module handles API key generation, validation, and user
creation for the server authentication layer.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.errors import ForgeDependencyError, ForgeServerError


def _import_sqlalchemy() -> Any:
    """Lazy-import sqlalchemy, raising on absence."""
    try:
        import sqlalchemy  # noqa: F811
        return sqlalchemy
    except ImportError as exc:
        raise ForgeDependencyError(
            "sqlalchemy is required for the collaboration server. "
            "Install it with: pip install sqlalchemy"
        ) from exc


def generate_api_key() -> str:
    """Generate a random API key using uuid4.

    Returns:
        A string API key in uuid4 format.
    """
    return str(uuid.uuid4())


def validate_api_key(api_key: str, session: Any) -> Any | None:
    """Look up an API key and return the associated user.

    Args:
        api_key: The API key to validate.
        session: SQLAlchemy session instance.

    Returns:
        The UserModel instance if found, or None.
    """
    _import_sqlalchemy()
    from server.models import UserModel
    return session.query(UserModel).filter(
        UserModel.api_key == api_key,
    ).first()


def require_api_key(api_key: str, session: Any) -> Any:
    """Validate an API key and raise if invalid.

    Args:
        api_key: The API key to validate.
        session: SQLAlchemy session instance.

    Returns:
        The authenticated UserModel instance.

    Raises:
        ForgeServerError: If the API key is invalid.
    """
    user = validate_api_key(api_key, session)
    if user is None:
        raise ForgeServerError(
            "Invalid API key: authentication failed"
        )
    return user


def create_user(username: str, session: Any) -> tuple[Any, str]:
    """Create a new user with a generated API key.

    Args:
        username: Unique username for the new user.
        session: SQLAlchemy session instance.

    Returns:
        Tuple of (UserModel instance, api_key string).

    Raises:
        ForgeServerError: If user creation fails.
    """
    _import_sqlalchemy()
    from server.models import UserModel
    api_key = generate_api_key()
    user = UserModel(username=username, api_key=api_key)
    try:
        session.add(user)
        session.commit()
        session.refresh(user)
    except Exception as exc:
        session.rollback()
        raise ForgeServerError(
            f"Failed to create user '{username}': {exc}"
        ) from exc
    return user, api_key
