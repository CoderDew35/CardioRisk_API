"""
CalculateRiskProfileUseCase

Core inference orchestrator. Runs when InferenceService consumes
a PatientTelemetryReceived event from RabbitMQ.

Pipeline:
  1. Load latest patient record from repository
  2. Retrieve features from FeatureStore (enforces DRY)
  3. Run LightGBM model via IRiskModel adapter
  4. Compute SHAP values via IExplainer adapter
  5. Generate clinical narrative via ILLMGateway (OpenRouter)
  6. Publish RiskScoreGenerated event to RabbitMQ
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from src.application.ports.interfaces import (
    IEventPublisher,
    IExplainer,
    IFeatureStore,
    ILLMGateway,
    IPatientRepository,
    IRiskModel,
)
from src.domain.events.cardiovascular_events import RiskScoreGenerated

logger = logging.getLogger(__name__)


class PatientNotFoundError(Exception):
    pass


class FeatureNotAvailableError(Exception):
    pass


@dataclass
class CalculateRiskProfileUseCase:
    """
    Dependency-injected inference pipeline. All dependencies are ports — 
    no infrastructure imports. Fully unit-testable with mocks.
    """
    patient_repository: IPatientRepository
    feature_store: IFeatureStore
    risk_model: IRiskModel
    explainer: IExplainer
    llm_gateway: ILLMGateway
    event_publisher: IEventPublisher

    async def execute(
        self,
        patient_id: UUID,
        time_step: int = 0,
        feature_overrides: dict[str, float] | None = None,
        is_counterfactual: bool = False,
        previous_shap: list | None = None,
    ) -> RiskScoreGenerated:
        """
        Args:
            patient_id:       Target patient UUID
            time_step:        0=baseline, 1–5=synthetic temporal step
            feature_overrides: For counterfactual simulations (what-if)
            is_counterfactual: Marks the event as a simulation result
            previous_shap:    Previous time step's contributions for Δ-SHAP

        Returns:
            RiskScoreGenerated domain event (also published to RabbitMQ)

        Raises:
            PatientNotFoundError: If patient has no records
            FeatureNotAvailableError: If feature store has no entry
        """
        # 1. Load patient record
        record = await self.patient_repository.get_latest_by_id(patient_id)
        if record is None:
            raise PatientNotFoundError(f"No record found for patient_id={patient_id}")

        # 2. Get features from FeatureStore (single source of truth)
        features = await self.feature_store.get_features(patient_id)
        if features is None:
            # Fallback: derive from entity directly (handles newly ingested records)
            features = record.to_feature_dict()
            logger.warning(
                "FeatureStore miss for patient_id=%s; using entity features", patient_id
            )

        # Apply overrides (counterfactual simulation)
        if feature_overrides:
            features = {**features, **feature_overrides}
            logger.info("Counterfactual overrides applied: %s", list(feature_overrides.keys()))

        # 3. Predict risk score
        risk_score = self.risk_model.predict(features)
        logger.info(
            "Risk predicted: patient_id=%s score=%.4f level=%s",
            patient_id, risk_score.value, risk_score.risk_level,
        )

        # 4. Compute SHAP values with optional delta from previous step
        contributions = self.explainer.explain(features, risk_score, previous_shap)

        # 5. Generate LLM clinical narrative
        patient_context = {
            "age_years": record.age_years,
            "gender": record.gender.name,
            "bmi": record.bmi,
            "ap_hi": record.ap_hi,
            "ap_lo": record.ap_lo,
            "bp_category": record.bp_category.value,
            "is_hypertensive": record.is_hypertensive,
            "time_step": time_step,
        }

        try:
            narrative = await self.llm_gateway.generate_narrative(
                patient_context=patient_context,
                shap_contributions=contributions,
                risk_score=risk_score,
            )
        except Exception as exc:
            # LLM failure is non-critical — system continues without narrative
            logger.error("LLM narrative generation failed: %s", exc)
            narrative = None

        # 6. Publish result event (skip for pure counterfactuals)
        event = RiskScoreGenerated(
            patient_id=patient_id,
            risk_score=risk_score.value,
            risk_level=risk_score.risk_level.value,
            shap_vector={c.feature_name: c.shap_value for c in contributions},
            llm_narrative=narrative,
            model_version=self.risk_model.model_version,
            time_step=time_step,
            is_counterfactual=is_counterfactual,
        )

        if not is_counterfactual:
            await self.event_publisher.publish(event)
            logger.info("Published RiskScoreGenerated: event_id=%s", event.event_id)

        return event
