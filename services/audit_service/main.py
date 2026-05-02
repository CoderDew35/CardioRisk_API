"""
AuditService — Compliance Tier Microservice

Consumes PatientTelemetryReceived events from RabbitMQ queue: audit.raw.q
Appends raw payload to Bronze Delta Lake table (immutable, append-only).
Publishes AuditLogWritten confirmation event.

This service NEVER modifies data. It is the system's compliance record.
Run: python services/audit_service/main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from src.infrastructure.messaging.consumer_base import BaseRabbitMQConsumer
from src.infrastructure.messaging.publisher import RabbitMQPublisher
from src.domain.events.cardiovascular_events import AuditLogWritten

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | AuditService | %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
BRONZE_PATH = os.getenv("DELTA_LAKE_PATH", "./data/lakehouse") + "/bronze/cardio_events"

BRONZE_SCHEMA = pa.schema([
    pa.field("patient_id",      pa.string()),
    pa.field("event_id",        pa.string()),
    pa.field("raw_payload",     pa.string()),   # JSON string
    pa.field("source",          pa.string()),
    pa.field("schema_version",  pa.string()),
    pa.field("ingested_at",     pa.timestamp("us", tz="UTC")),
])


class AuditConsumer(BaseRabbitMQConsumer):

    def __init__(self, publisher: RabbitMQPublisher) -> None:
        super().__init__(
            rabbitmq_url=RABBITMQ_URL,
            queue_name="audit.raw.q",
            routing_key="patient.telemetry.raw",
            prefetch_count=5,
        )
        self._publisher = publisher

    async def process_message(self, body: dict[str, Any]) -> None:
        patient_id = body.get("patient_id", "unknown")
        event_id = body.get("event_id", str(uuid.uuid4()))

        # Append to Bronze Delta (immutable, never overwrite)
        record = {
            "patient_id":     patient_id,
            "event_id":       event_id,
            "raw_payload":    json.dumps(body.get("raw_payload", {})),
            "source":         body.get("source", "unknown"),
            "schema_version": body.get("schema_version", "1.0"),
            "ingested_at":    datetime.now(timezone.utc),
        }

        table = pa.Table.from_pylist([record], schema=BRONZE_SCHEMA)

        write_deltalake(
            BRONZE_PATH,
            table,
            mode="append",
            schema_mode="merge",
        )

        logger.info("Bronze append: patient_id=%s event_id=%s", patient_id, event_id)

        # Publish compliance acknowledgement
        audit_event = AuditLogWritten(
            patient_id=uuid.UUID(patient_id),
            original_event_id=uuid.UUID(event_id),
            bronze_path=BRONZE_PATH,
        )
        await self._publisher.publish(audit_event)


async def main() -> None:
    logger.info("AuditService starting...")
    publisher = RabbitMQPublisher()
    await publisher.connect()

    consumer = AuditConsumer(publisher=publisher)
    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await consumer.stop()
        await publisher.close()
        logger.info("AuditService stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
