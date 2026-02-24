"""Unit tests for WebSocket connection manager."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from server.websocket_manager import ConnectionManager, notify_run_update


def _make_ws() -> AsyncMock:
    """Create a mock WebSocket with accept and send_text."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


def _run(coro):  # type: ignore[no-untyped-def]
    """Run a coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_connect_adds_websocket() -> None:
    """connect() should accept and register the WebSocket."""
    manager = ConnectionManager()
    ws = _make_ws()

    _run(manager.connect(ws))

    assert ws in manager.active_connections
    ws.accept.assert_awaited_once()


def test_disconnect_removes_websocket() -> None:
    """disconnect() should remove the WebSocket from active list."""
    manager = ConnectionManager()
    ws = _make_ws()

    _run(manager.connect(ws))
    manager.disconnect(ws)

    assert ws not in manager.active_connections


def test_broadcast_sends_to_all() -> None:
    """broadcast() should send message to all active connections."""
    manager = ConnectionManager()
    ws1 = _make_ws()
    ws2 = _make_ws()

    async def _setup_and_broadcast() -> None:
        await manager.connect(ws1)
        await manager.connect(ws2)
        await manager.broadcast("hello")

    _run(_setup_and_broadcast())

    ws1.send_text.assert_awaited_once_with("hello")
    ws2.send_text.assert_awaited_once_with("hello")


def test_notify_run_update_broadcasts_json() -> None:
    """notify_run_update should broadcast a JSON payload."""
    manager = ConnectionManager()
    ws = _make_ws()

    async def _setup_and_notify() -> None:
        await manager.connect(ws)
        await notify_run_update(manager, "run-123", "completed")

    _run(_setup_and_notify())

    call_args = ws.send_text.call_args[0][0]
    payload = json.loads(call_args)
    assert payload["type"] == "run_update"
    assert payload["run_id"] == "run-123"
    assert payload["status"] == "completed"


def test_disconnect_nonexistent_is_noop() -> None:
    """disconnect() on an unregistered WebSocket should be a no-op."""
    manager = ConnectionManager()
    ws = _make_ws()
    manager.disconnect(ws)
    assert len(manager.active_connections) == 0
