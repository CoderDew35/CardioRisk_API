"""
InferenceService — Clinical Tier Microservice

Consumes PatientTelemetryReceived events from RabbitMQ queue: inference.raw.q
Runs the full ML → SHAP → LLM → Gold Delta pipeline.
Publishes RiskScoreGenerated events.

Run: python services/inference_service/main.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from src.infrastructure.messaging.consumer_base import BaseRabbitMQConsumer
from src.infrastructure.messaging.publisher import RabbitMQPublisher
from src.infrastructure.llm.openrouter_gateway import OpenRouterGateway
from src.infrastructure.ml.lightgbm_adapter import LightGBMAdapter
from src.infrastructure.ml.shap_adapter import SHAPTreeExplainerAdapter
from src.application.use_cases.calculate_risk_profile import CalculateRiskProfileUseCase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | InferenceService | %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


class InferenceConsumer(BaseRabbitMQConsumer):

    def __init__(
        self,
        publisher: RabbitMQPublisher,
        session_factory,
        model: LightGBMAdapter,
        explainer: SHAPTreeExplainerAdapter,
        llm: OpenRouterGateway,
    ) -> None:
        super().__init__(
            rabbitmq_url=RABBITMQ_URL,
            queue_name="inference.raw.q",
            routing_key="patient.telemetry.raw",
            prefetch_count=3,   # Lower prefetch — inference is CPU-bound
        )
        self._publisher = publisher
        self._session_factory = session_factory
        self._model = model
        self._explainer = explainer
        self._llm = llm

    async def process_message(self, body: dict[str, Any]) -> None:
        patient_id_str = body.get("patient_id")
        if not patient_id_str:
            raise ValueError("Message missing patient_id")

        patient_id = uuid.UUID(patient_id_str)
        logger.info("Processing inference for patient_id=%s", patient_id)

        # Fresh session per message — prevents stale state accumulation
        from src.infrastructure.db.patient_repository import PostgreSQLPatientRepository
        from src.infrastructure.delta.feature_store import DeltaFeatureStore

        async with self._session_factory() as session:
            use_case = CalculateRiskProfileUseCase(
                patient_repository=PostgreSQLPatientRepository(session),
                feature_store=DeltaFeatureStore(),
                risk_model=self._model,
                explainer=self._explainer,
                llm_gateway=self._llm,
                event_publisher=self._publisher,
            )
            event = await use_case.execute(patient_id=patient_id)
            await session.commit()

        logger.info(
            "Inference complete: patient_id=%s risk=%.4f level=%s",
            patient_id, event.risk_score, event.risk_level,
        )


async def main() -> None:
    logger.info("InferenceService starting...")

    # Bootstrap adapters
    model = LightGBMAdapter()
    explainer = SHAPTreeExplainerAdapter(model=model)
    llm = OpenRouterGateway()
    publisher = RabbitMQPublisher()
    await publisher.connect()

    from src.infrastructure.db.database import AsyncSessionFactory

    consumer = InferenceConsumer(
        publisher=publisher,
        session_factory=AsyncSessionFactory,
        model=model,
        explainer=explainer,
        llm=llm,
    )

    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await consumer.stop()
        await publisher.close()
        logger.info("InferenceService stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())

