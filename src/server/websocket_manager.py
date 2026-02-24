"""WebSocket connection manager for Forge collaboration server.

This module manages active WebSocket connections and provides
broadcast capabilities for real-time run update notifications.
"""

from __future__ import annotations

import json
from typing import Any


class ConnectionManager:
    """Manages active WebSocket connections for broadcast.

    Attributes:
        active_connections: List of currently connected WebSockets.
    """

    def __init__(self) -> None:
        """Initialize with empty connection list."""
        self.active_connections: list[Any] = []

    async def connect(self, websocket: Any) -> None:
        """Accept and register a WebSocket connection.

        Args:
            websocket: The WebSocket instance to register.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: Any) -> None:
        """Remove a WebSocket connection from active list.

        Args:
            websocket: The WebSocket instance to remove.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        """Send a message to all active connections.

        Args:
            message: The text message to broadcast.
        """
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)


async def notify_run_update(
    manager: ConnectionManager,
    run_id: str,
    status: str,
) -> None:
    """Broadcast a run status update to all connected clients.

    Args:
        manager: The WebSocket connection manager.
        run_id: Identifier of the updated run.
        status: New status of the run.
    """
    payload = json.dumps({
        "type": "run_update",
        "run_id": run_id,
        "status": status,
    })
    await manager.broadcast(payload)
