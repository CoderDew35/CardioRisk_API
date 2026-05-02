"""
FastAPI Application — main entry point.

Routers: /v1/patients, /v1/cohort
WebSocket: /v1/patients/{id}/live
Lifespan: RabbitMQ publisher connect/close, WebSocket manager, dashboard consumer
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import aio_pika
from aio_pika import ExchangeType
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.interfaces.api.routers import patients, cohort, health
from src.infrastructure.messaging.publisher import RabbitMQPublisher
from src.infrastructure.messaging.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
EXCHANGE_NAME = "cardiorisk.events"

# Singletons
_publisher = RabbitMQPublisher()
_ws_manager = WebSocketManager()


async def _run_dashboard_consumer(manager: WebSocketManager) -> None:
    """Background task: consume RiskScoreGenerated events and relay to WebSockets."""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            EXCHANGE_NAME, ExchangeType.TOPIC, durable=True
        )
        queue = await channel.declare_queue(
            "dashboard.scores.q", durable=True
        )
        await queue.bind(exchange, routing_key="risk.score.generated")

        logger.info("Dashboard consumer started — relaying to WebSockets")

        async with queue.iterator() as messages:
            async for message in messages:
                try:
                    body = json.loads(message.body)
                    patient_id = body.get("patient_id", "")
                    await manager.broadcast(patient_id, body)
                    await message.ack()
                except Exception as exc:
                    logger.warning("Dashboard relay error: %s", exc)
                    await message.reject(requeue=False)
    except Exception as exc:
        logger.error("Dashboard consumer failed to start: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect to RabbitMQ, start dashboard consumer. Shutdown: close."""
    logger.info("CardioRisk API starting up...")
    await _publisher.connect()
    app.state.publisher = _publisher
    app.state.ws_manager = _ws_manager

    # Start dashboard consumer as background task
    consumer_task = asyncio.create_task(_run_dashboard_consumer(_ws_manager))

    yield

    logger.info("CardioRisk API shutting down...")
    consumer_task.cancel()
    await _publisher.close()


app = FastAPI(
    title="CardioRisk XAI API",
    description=(
        "Explainable AI backend for personalized cardiovascular risk trajectories. "
        "Provides temporal SHAP analysis, counterfactual simulations, and "
        "GenAI clinical narratives."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA / Next.js dev server
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(patients.router, prefix="/v1/patients", tags=["Patients"])
app.include_router(cohort.router, prefix="/v1/cohort", tags=["Cohort"])
