"""
Application Layer Port Interfaces — abstract contracts for all adapters.

These are Python Protocols (structural typing). Any class implementing
the required methods satisfies the interface — no explicit inheritance needed.
This pattern enables full mock injection in unit tests without any real infrastructure.

Clean Architecture rule: Application layer NEVER imports from infrastructure.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from src.domain.entities.patient_cardiovascular_record import PatientCardiovascularRecord
from src.domain.events.cardiovascular_events import (
    AuditLogWritten,
    PatientTelemetryReceived,
    RiskScoreGenerated,
)
from src.domain.value_objects.risk_trajectory import (
    RiskScore,
    RiskTrajectoryPoint,
    SHAPContribution,
)


#IPatientRepository ────

@runtime_checkable
class IPatientRepository(Protocol):
    """Persist and retrieve PatientCardiovascularRecord from PostgreSQL."""

    async def save(self, record: PatientCardiovascularRecord) -> None: ...

    async def get_by_id(self, patient_id: UUID) -> PatientCardiovascularRecord | None: ...

    async def get_latest_by_id(self, patient_id: UUID) -> PatientCardiovascularRecord | None: ...

    async def get_trajectory_records(
        self, patient_id: UUID, limit: int = 10
    ) -> list[PatientCardiovascularRecord]: ...


#IFeatureStore ─────────

@runtime_checkable
class IFeatureStore(Protocol):
    """
    Centralized feature retrieval (enforces DRY — single source of truth).
    Reads from Silver Delta layer. Used by both services identically.
    """

    async def get_features(self, patient_id: UUID) -> dict[str, float] | None: ...

    async def get_features_bulk(
        self, patient_ids: list[UUID]
    ) -> dict[UUID, dict[str, float]]: ...


#IRiskModel ────────────

@runtime_checkable
class IRiskModel(Protocol):
    """Predict cardiovascular risk probability from a feature dict."""

    def predict(self, features: dict[str, float]) -> RiskScore: ...

    def predict_proba(self, features: dict[str, float]) -> float: ...

    @property
    def model_version(self) -> str: ...

    @property
    def feature_names(self) -> list[str]: ...


#IExplainer ────────────

@runtime_checkable
class IExplainer(Protocol):
    """Compute SHAP values for a prediction."""

    def explain(
        self,
        features: dict[str, float],
        risk_score: RiskScore,
        previous_contributions: list[SHAPContribution] | None = None,
    ) -> list[SHAPContribution]: ...


#ILLMGateway ───────────

@runtime_checkable
class ILLMGateway(Protocol):
    """Generate clinical narrative explanations from SHAP data via OpenRouter."""

    async def generate_narrative(
        self,
        patient_context: dict[str, Any],
        shap_contributions: list[SHAPContribution],
        risk_score: RiskScore,
        delta_score: float | None = None,
    ) -> str: ...


#IEventPublisher ───────

@runtime_checkable
class IEventPublisher(Protocol):
    """Publish domain events to RabbitMQ exchange."""

    async def publish(
        self,
        event: PatientTelemetryReceived | RiskScoreGenerated | AuditLogWritten,
    ) -> None: ...


#IEventConsumer ────────

@runtime_checkable
class IEventConsumer(Protocol):
    """Consume events from a RabbitMQ queue (blocking loop)."""

    async def start_consuming(self) -> None: ...

    async def stop(self) -> None: ...


#IDeltaWriter ──────────

@runtime_checkable
class IDeltaWriter(Protocol):
    """Write records to a Delta Lake table (Bronze / Silver / Gold)."""

    async def append(self, records: list[dict[str, Any]]) -> None: ...

    async def upsert(
        self, records: list[dict[str, Any]], merge_keys: list[str]
    ) -> None: ...


#ICohortRepository ─────

@runtime_checkable
class ICohortRepository(Protocol):
    """Read cohort-level aggregates from the Gold Delta layer."""

    async def get_age_cohort_stats(
        self, age_min: int, age_max: int
    ) -> dict[str, Any]: ...

    async def get_risk_distribution(self) -> dict[str, float]: ...

    async def get_patient_percentile(
        self, patient_id: UUID
    ) -> float | None: ...
