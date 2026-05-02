"""
WebSocket Manager — relays RabbitMQ events to connected dashboard clients.

Architecture:
  RabbitMQ (risk.score.generated) → DashboardConsumer → WebSocketManager → Browser

The manager maintains a registry of connected WebSocket clients keyed by patient_id.
When a RiskScoreGenerated event arrives, it broadcasts to all clients watching that patient.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Thread-safe registry of WebSocket connections per patient_id."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, patient_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[patient_id].add(websocket)
        logger.info("WebSocket connected: patient_id=%s (total=%d)",
                     patient_id, len(self._connections[patient_id]))

    def disconnect(self, patient_id: str, websocket: WebSocket) -> None:
        self._connections[patient_id].discard(websocket)
        if not self._connections[patient_id]:
            del self._connections[patient_id]
        logger.info("WebSocket disconnected: patient_id=%s", patient_id)

    async def broadcast(self, patient_id: str, data: dict) -> None:
        """Send event data to all clients watching this patient."""
        clients = self._connections.get(patient_id, set())
        if not clients:
            return

        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[patient_id].discard(ws)

    async def broadcast_all(self, data: dict) -> None:
        """Send event data to ALL connected clients (e.g., system alerts)."""
        for patient_id in list(self._connections.keys()):
            await self.broadcast(patient_id, data)

    @property
    def active_connections(self) -> int:
        return sum(len(s) for s in self._connections.values())
