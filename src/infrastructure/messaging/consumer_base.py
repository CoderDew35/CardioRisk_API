"""
RabbitMQ Consumer Base — abstract base for AuditService and InferenceService workers.

Features:
  - Manual acknowledgement (at-least-once delivery)
  - Exponential backoff retry (3 attempts)
  - Dead Letter Queue (DLQ) routing on permanent failure
  - Graceful shutdown via asyncio Event
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import aio_pika
from aio_pika import ExchangeType

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "cardiorisk.events"
DLX_NAME = "cardiorisk.events.dlx"   # Dead Letter Exchange
MAX_RETRIES = 3


class BaseRabbitMQConsumer(ABC):
    """
    Subclass this and implement process_message() for each service.

    Usage:
        class AuditConsumer(BaseRabbitMQConsumer):
            async def process_message(self, body: dict) -> None:
                ...

        consumer = AuditConsumer(
            rabbitmq_url="amqp://...",
            queue_name="audit.raw.q",
            routing_key="patient.telemetry.raw",
        )
        await consumer.start_consuming()
    """

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str,
        routing_key: str,
        prefetch_count: int = 10,
    ) -> None:
        self._rabbitmq_url = rabbitmq_url
        self._queue_name = queue_name
        self._routing_key = routing_key
        self._prefetch_count = prefetch_count
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._should_stop = asyncio.Event()

    @abstractmethod
    async def process_message(self, body: dict[str, Any]) -> None:
        """
        Override in each service. Raise an exception to trigger retry/DLQ.
        """
        ...

    async def start_consuming(self) -> None:
        self._connection = await aio_pika.connect_robust(self._rabbitmq_url)
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=self._prefetch_count)

        # Declare main exchange
        exchange = await channel.declare_exchange(
            EXCHANGE_NAME, ExchangeType.TOPIC, durable=True
        )

        # Declare Dead Letter Exchange
        dlx = await channel.declare_exchange(
            DLX_NAME, ExchangeType.FANOUT, durable=True
        )
        dlq = await channel.declare_queue(
            f"{self._queue_name}.dlq", durable=True
        )
        await dlq.bind(dlx)

        # Declare main queue with DLX routing
        queue = await channel.declare_queue(
            self._queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": DLX_NAME,
            },
        )
        await queue.bind(exchange, routing_key=self._routing_key)

        logger.info(
            "Consumer started: queue=%s routing_key=%s",
            self._queue_name, self._routing_key,
        )

        async with queue.iterator() as messages:
            async for message in messages:
                if self._should_stop.is_set():
                    break

                await self._handle_with_retry(message)

    async def _handle_with_retry(
        self, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        body: dict[str, Any] = {}
        attempt = 0

        try:
            body = json.loads(message.body)
        except json.JSONDecodeError as exc:
            logger.error("Malformed message body — sending to DLQ: %s", exc)
            await message.reject(requeue=False)
            return

        while attempt < MAX_RETRIES:
            try:
                await self.process_message(body)
                await message.ack()
                return
            except Exception as exc:
                attempt += 1
                wait = 2 ** attempt  # 2s, 4s, 8s
                logger.warning(
                    "Message processing failed (attempt %d/%d): %s. Retrying in %ds",
                    attempt, MAX_RETRIES, exc, wait,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

        # All retries exhausted → reject to DLQ
        logger.error(
            "Max retries exhausted for message_id=%s — routing to DLQ",
            message.message_id,
        )
        await message.reject(requeue=False)

    async def stop(self) -> None:
        self._should_stop.set()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("Consumer stopped: queue=%s", self._queue_name)
