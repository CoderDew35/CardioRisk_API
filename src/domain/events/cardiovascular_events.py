"""
Domain Events — immutable event records published to RabbitMQ.

All events are frozen dataclasses with zero external dependencies.
Each event carries enough context to be processed independently
(self-describing payload).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#PatientTelemetryReceived ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatientTelemetryReceived:
    """
    Published when a raw patient record is ingested via the API.
    Consumed by: AuditService, InferenceService.
    Routing key: patient.telemetry.raw
    """
    patient_id: UUID
    raw_payload: dict          # Exact fields as received (before validation)
    source: str                # "api" | "batch_csv" | "websocket"
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utcnow)
    schema_version: str = "1.0"

    @property
    def routing_key(self) -> str:
        return "patient.telemetry.raw"


#RiskScoreGenerated ────

@dataclass(frozen=True)
class RiskScoreGenerated:
    """
    Published by InferenceService after model prediction + SHAP + LLM.
    Consumed by: Dashboard WebSocket relay, future notification service.
    Routing key: risk.score.generated
    """
    patient_id: UUID
    risk_score: float                 # 0.0–1.0
    risk_level: str                   # "Low" | "Moderate" | "High" | "Very High"
    shap_vector: dict[str, float]     # {feature_name: shap_value}
    llm_narrative: str | None
    model_version: str
    time_step: int = 0                # 0=baseline, 1–5=synthetic temporal
    is_counterfactual: bool = False
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utcnow)

    @property
    def routing_key(self) -> str:
        return "risk.score.generated"


#AuditLogWritten ───────

@dataclass(frozen=True)
class AuditLogWritten:
    """
    Published by AuditService after successfully persisting to Bronze Delta.
    Compliance acknowledgement — confirms immutable storage.
    Routing key: audit.log.written
    """
    patient_id: UUID
    original_event_id: UUID      # ID of the PatientTelemetryReceived event
    bronze_path: str             # Delta table path where record was appended
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utcnow)

    @property
    def routing_key(self) -> str:
        return "audit.log.written"


# ── ModelDriftDetected ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelDriftDetected:
    """
    Published by DriftDetectionService when statistical divergence is detected.
    Consumed by: ContinuousTrainingService.
    Routing key: model.drift.detected
    """
    drifted_features: dict[str, float]   # {feature_name: ks_statistic}
    psi_scores: dict[str, float]         # {feature_name: psi_value}
    window_size: int
    current_model_version: str
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utcnow)

    @property
    def routing_key(self) -> str:
        return "model.drift.detected"


# ── ModelRetrained ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelRetrained:
    """
    Published by ContinuousTrainingService after retrain + evaluation.
    Consumed by: API (hot-swap trigger), Dashboard (notification).
    Routing key: model.retrained
    """
    new_model_version: str
    old_model_version: str
    auc_roc_new: float
    auc_roc_old: float
    promoted: bool                       # True if new model replaced old
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utcnow)

    @property
    def routing_key(self) -> str:
        return "model.retrained"

