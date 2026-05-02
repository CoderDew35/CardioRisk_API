"""
RunCounterfactualUseCase

What-if simulation engine: applies user-defined feature deltas to a patient's
baseline, re-runs prediction + SHAP, and returns the new trajectory WITHOUT
publishing any events (pure read/compute operation).

Used by the dashboard counterfactual explorer:
  e.g., "What happens to risk if ap_hi drops from 160 → 130?"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from src.application.ports.interfaces import (
    IExplainer,
    IFeatureStore,
    ILLMGateway,
    IPatientRepository,
    IRiskModel,
)
from src.domain.value_objects.risk_trajectory import RiskScore, RiskTrajectoryPoint

logger = logging.getLogger(__name__)

COUNTERFACTUAL_LABELS = {
    "smoke":              "Stop Smoking",
    "active":             "Increase Physical Activity",
    "ap_hi":              "Reduce Systolic BP",
    "weight_kg":          "Weight Reduction",
    "bmi":                "BMI Reduction",
    "drinks_alcohol":     "Reduce Alcohol",
    "cholesterol":        "Lower Cholesterol",
}


@dataclass
class RunCounterfactualUseCase:
    """
    Computes counterfactual risk trajectory — no side effects, no events.
    Returns both the baseline and counterfactual RiskTrajectoryPoint for comparison.
    """
    patient_repository: IPatientRepository
    feature_store: IFeatureStore
    risk_model: IRiskModel
    explainer: IExplainer
    llm_gateway: ILLMGateway

    async def execute(
        self,
        patient_id: UUID,
        feature_overrides: dict[str, float],
        include_narrative: bool = True,
    ) -> dict:
        """
        Args:
            patient_id:        Target patient
            feature_overrides: Feature deltas to apply {feature_name: new_value}
            include_narrative: Whether to generate LLM explanation of the change

        Returns:
            {
              "baseline": RiskTrajectoryPoint,
              "counterfactual": RiskTrajectoryPoint,
              "risk_delta": float,          # counterfactual - baseline
              "label": str,                 # human-readable intervention name
            }
        """
        record = await self.patient_repository.get_latest_by_id(patient_id)
        if record is None:
            raise ValueError(f"No record for patient_id={patient_id}")

        baseline_features = await self.feature_store.get_features(patient_id)
        if baseline_features is None:
            baseline_features = record.to_feature_dict()

        #Baseline ────
        baseline_score = self.risk_model.predict(baseline_features)
        baseline_contributions = self.explainer.explain(baseline_features, baseline_score)
        baseline_point = RiskTrajectoryPoint(
            time_step=0,
            timestamp=datetime.now(timezone.utc),
            risk_score=baseline_score,
            shap_contributions=tuple(baseline_contributions),
            is_counterfactual=False,
        )

        #Counterfactual ────────────────────────────────────────────────────
        cf_features = {**baseline_features, **feature_overrides}
        cf_score = self.risk_model.predict(cf_features)
        cf_contributions = self.explainer.explain(cf_features, cf_score, baseline_contributions)

        # Build label from changed features
        changed_keys = list(feature_overrides.keys())
        label = COUNTERFACTUAL_LABELS.get(changed_keys[0], "Intervention") if changed_keys else "Intervention"

        # Generate comparative narrative
        narrative = None
        if include_narrative:
            try:
                narrative = await self.llm_gateway.generate_narrative(
                    patient_context={
                        "age_years": record.age_years,
                        "gender": record.gender.name,
                        "intervention": label,
                        "feature_overrides": feature_overrides,
                    },
                    shap_contributions=cf_contributions,
                    risk_score=cf_score,
                    delta_score=cf_score.delta(baseline_score),
                )
            except Exception as exc:
                logger.warning("LLM counterfactual narrative failed: %s", exc)

        cf_point = RiskTrajectoryPoint(
            time_step=1,
            timestamp=datetime.now(timezone.utc),
            risk_score=cf_score,
            shap_contributions=tuple(cf_contributions),
            llm_narrative=narrative,
            is_counterfactual=True,
            counterfactual_label=label,
        )

        risk_delta = cf_score.delta(baseline_score)
        logger.info(
            "Counterfactual '%s': risk %.4f → %.4f (Δ=%.4f) for patient_id=%s",
            label, baseline_score.value, cf_score.value, risk_delta, patient_id,
        )

        return {
            "baseline": baseline_point,
            "counterfactual": cf_point,
            "risk_delta": risk_delta,
            "label": label,
        }
