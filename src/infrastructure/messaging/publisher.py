"""
RabbitMQ Event Publisher — implements IEventPublisher port.

Uses aio-pika for async AMQP publishing.
Exchange: cardiorisk.events (topic type)
All messages published as persistent (delivery_mode=2) for durability.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from src.domain.events.cardiovascular_events import (
    AuditLogWritten,
    ModelDriftDetected,
    ModelRetrained,
    PatientTelemetryReceived,
    RiskScoreGenerated,
)

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "cardiorisk.events"
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

DomainEvent = (
    PatientTelemetryReceived | RiskScoreGenerated | AuditLogWritten
    | ModelDriftDetected | ModelRetrained
)


def _serialise_event(event: DomainEvent) -> dict[str, Any]:
    """Convert domain event to JSON-serialisable dict."""
    base = {
        "event_id": str(event.event_id),
        "event_type": type(event).__name__,
        "occurred_at": event.occurred_at.isoformat(),
    }

    # Patient-scoped events carry patient_id
    if hasattr(event, "patient_id"):
        base["patient_id"] = str(event.patient_id)

    if isinstance(event, PatientTelemetryReceived):
        base.update({
            "raw_payload": event.raw_payload,
            "source": event.source,
            "schema_version": event.schema_version,
        })
    elif isinstance(event, RiskScoreGenerated):
        base.update({
            "risk_score": event.risk_score,
            "risk_level": event.risk_level,
            "shap_vector": event.shap_vector,
            "llm_narrative": event.llm_narrative,
            "model_version": event.model_version,
            "time_step": event.time_step,
            "is_counterfactual": event.is_counterfactual,
        })
    elif isinstance(event, AuditLogWritten):
        base.update({
            "original_event_id": str(event.original_event_id),
            "bronze_path": event.bronze_path,
        })
    elif isinstance(event, ModelDriftDetected):
        base.update({
            "drifted_features": event.drifted_features,
            "psi_scores": event.psi_scores,
            "window_size": event.window_size,
            "current_model_version": event.current_model_version,
        })
    elif isinstance(event, ModelRetrained):
        base.update({
            "new_model_version": event.new_model_version,
            "old_model_version": event.old_model_version,
            "auc_roc_new": event.auc_roc_new,
            "auc_roc_old": event.auc_roc_old,
            "promoted": event.promoted,
        })

    return base


class RabbitMQPublisher:
    """Async event publisher. Call connect() before first publish."""

    def __init__(self) -> None:
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(RABBITMQ_URL)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            EXCHANGE_NAME,
            ExchangeType.TOPIC,
            durable=True,
        )
        logger.info("RabbitMQ publisher connected to exchange: %s", EXCHANGE_NAME)

    async def publish(self, event: DomainEvent) -> None:
        if self._exchange is None:
            raise RuntimeError("Publisher not connected. Call connect() first.")

        payload = _serialise_event(event)
        body = json.dumps(payload, default=str).encode()

        # Correlation ID: patient_id for patient-scoped events, event_id otherwise
        correlation_id = (
            str(event.patient_id) if hasattr(event, "patient_id")
            else str(event.event_id)
        )

        message = Message(
            body=body,
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,  # survives broker restart
            message_id=str(event.event_id),
            correlation_id=correlation_id,
        )

        await self._exchange.publish(
            message=message,
            routing_key=event.routing_key,
        )
        logger.debug(
            "Published %s [routing_key=%s, message_id=%s]",
            type(event).__name__, event.routing_key, event.event_id,
        )

    async def close(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("RabbitMQ publisher connection closed")
