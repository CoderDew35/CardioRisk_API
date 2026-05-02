"""
IngestPatientTelemetryUseCase

Orchestrates the ingestion of a raw patient payload:
  1. Validates raw fields (FeatureValidator domain service)
  2. Derives bmi, age_years, bp_category (domain services)
  3. Constructs PatientCardiovascularRecord entity
  4. Persists to PostgreSQL via IPatientRepository
  5. Publishes PatientTelemetryReceived event to RabbitMQ
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from src.domain.entities.enums import (
    BPCategory,
    CholesterolLevel,
    Gender,
    GlucoseLevel,
)
from src.domain.entities.patient_cardiovascular_record import PatientCardiovascularRecord
from src.domain.events.cardiovascular_events import PatientTelemetryReceived
from src.domain.services.bp_classifier import BPClassifier
from src.domain.services.feature_validator import FeatureValidator
from src.application.ports.interfaces import IEventPublisher, IPatientRepository

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when payload fails clinical validation."""
    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


@dataclass
class IngestPatientTelemetryUseCase:
    """
    Single-responsibility: validate → construct entity → persist → publish.

    Dependencies injected via ports — no infrastructure imports here.
    """
    patient_repository: IPatientRepository
    event_publisher: IEventPublisher

    async def execute(
        self,
        raw_payload: dict[str, Any],
        patient_id: UUID | None = None,
        source: str = "api",
    ) -> PatientCardiovascularRecord:
        """
        Args:
            raw_payload: Dict matching dataset column names
            patient_id:  Optional UUID; generated if not provided
            source:      "api" | "batch_csv" | "websocket"

        Returns:
            Constructed and persisted PatientCardiovascularRecord

        Raises:
            ValidationError: If clinical validation fails
        """
        # 1. Validate raw payload
        result = FeatureValidator.validate(raw_payload)
        if not result.is_valid:
            logger.warning("Validation failed for payload: %s", result.errors)
            raise ValidationError(result.errors)

        if result.warnings:
            for warning in result.warnings:
                logger.warning("Clinical warning: %s", warning)

        # 2. Derive fields
        pid = patient_id or uuid4()
        age_days = int(raw_payload["age"])
        age_years = round(age_days / 365.25, 2)
        height_cm = int(raw_payload["height"])
        weight_kg = float(raw_payload["weight"])
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 4)

        ap_hi = int(raw_payload["ap_hi"])
        ap_lo = int(raw_payload["ap_lo"])

        # Use dataset's pre-encoded bp_category if available, else derive
        if "bp_category_encoded" in raw_payload and raw_payload["bp_category_encoded"] is not None:
            bp_category = BPClassifier.from_encoded(int(raw_payload["bp_category_encoded"]))
        else:
            bp_category = BPClassifier.classify(ap_hi, ap_lo)

        # 3. Construct immutable domain entity
        record = PatientCardiovascularRecord(
            patient_id=pid,
            recorded_at=datetime.now(timezone.utc),
            age_days=age_days,
            gender=Gender(int(raw_payload["gender"])),
            height_cm=height_cm,
            weight_kg=weight_kg,
            ap_hi=ap_hi,
            ap_lo=ap_lo,
            cholesterol=CholesterolLevel(int(raw_payload["cholesterol"])),
            glucose=GlucoseLevel(int(raw_payload["gluc"])),
            is_smoker=bool(int(raw_payload["smoke"])),
            drinks_alcohol=bool(int(raw_payload["alco"])),
            is_physically_active=bool(int(raw_payload["active"])),
            age_years=age_years,
            bmi=bmi,
            bp_category=bp_category,
            has_cardiovascular_disease=(
                bool(int(raw_payload["cardio"])) if "cardio" in raw_payload else None
            ),
        )

        # 4. Persist to PostgreSQL
        await self.patient_repository.save(record)
        logger.info("Saved patient record: patient_id=%s", pid)

        # 5. Publish domain event to RabbitMQ
        event = PatientTelemetryReceived(
            patient_id=pid,
            raw_payload=raw_payload,
            source=source,
        )
        await self.event_publisher.publish(event)
        logger.info("Published PatientTelemetryReceived: event_id=%s", event.event_id)

        return record
