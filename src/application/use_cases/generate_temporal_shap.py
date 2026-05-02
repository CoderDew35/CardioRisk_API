"""
GenerateTemporalSHAPUseCase

Generates a patient's risk trajectory across T time steps using
Monte Carlo synthetic perturbations of their baseline record.

This is the thesis's core novelty: Temporal SHAP (Δ-SHAP over time)
showing which features are driving risk change month-over-month.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from src.application.ports.interfaces import (
    IExplainer,
    IFeatureStore,
    ILLMGateway,
    IPatientRepository,
    IRiskModel,
)
from src.domain.value_objects.risk_trajectory import RiskTrajectoryPoint, RiskScore

logger = logging.getLogger(__name__)

# Monte Carlo perturbation parameters (one synthetic step ≈ one month)
# Only ROOT features are perturbed; derived features (bmi, pulse_pressure,
# mean_arterial_pressure) are re-computed from roots to maintain consistency.
_PERTURBATION_CONFIG: dict[str, dict] = {
    "ap_hi":             {"std": 3.0,  "min": 60,  "max": 250},
    "ap_lo":             {"std": 2.0,  "min": 40,  "max": 200},
    "weight_kg":         {"std": 0.5,  "min": 30,  "max": 200},
}


@dataclass
class GenerateTemporalSHAPUseCase:
    """
    Produces list[RiskTrajectoryPoint] for dashboard risk trajectory charts.

    Each step uses Monte Carlo perturbation to simulate realistic month-to-month
    physiological drift. Δ-SHAP values show which features are worsening/improving.
    """
    patient_repository: IPatientRepository
    feature_store: IFeatureStore
    risk_model: IRiskModel
    explainer: IExplainer
    llm_gateway: ILLMGateway

    async def execute(
        self,
        patient_id: UUID,
        n_steps: int = 5,
        seed: int | None = None,
        include_narratives: bool = True,
    ) -> list[RiskTrajectoryPoint]:
        """
        Args:
            patient_id:         Target patient
            n_steps:            Number of future synthetic time steps (default 5)
            seed:               Random seed for reproducibility
            include_narratives: Whether to call LLM for each step

        Returns:
            List of RiskTrajectoryPoint, length = n_steps + 1 (T=0 baseline included)
        """
        if seed is not None:
            random.seed(seed)

        # Load baseline features
        record = await self.patient_repository.get_latest_by_id(patient_id)
        if record is None:
            raise ValueError(f"No record found for patient_id={patient_id}")

        features = await self.feature_store.get_features(patient_id)
        if features is None:
            features = record.to_feature_dict()

        trajectory: list[RiskTrajectoryPoint] = []
        current_features = dict(features)
        previous_contributions = None
        base_time = datetime.now(timezone.utc)

        for step in range(n_steps + 1):  # 0 = baseline
            timestamp = base_time + timedelta(days=30 * step)

            # Predict and explain
            risk_score = self.risk_model.predict(current_features)
            contributions = self.explainer.explain(
                current_features, risk_score, previous_contributions
            )

            # Generate narrative (only baseline + last step unless all requested)
            narrative = None
            if include_narratives and (step == 0 or step == n_steps):
                try:
                    narrative = await self.llm_gateway.generate_narrative(
                        patient_context={
                            "age_years": record.age_years,
                            "gender": record.gender.name,
                            "bmi": current_features.get("bmi", record.bmi),
                            "ap_hi": current_features.get("ap_hi", record.ap_hi),
                            "ap_lo": current_features.get("ap_lo", record.ap_lo),
                            "time_step": step,
                        },
                        shap_contributions=contributions,
                        risk_score=risk_score,
                    )
                except Exception as exc:
                    logger.warning("LLM failed at step %d: %s", step, exc)

            trajectory.append(
                RiskTrajectoryPoint(
                    time_step=step,
                    timestamp=timestamp,
                    risk_score=risk_score,
                    shap_contributions=tuple(contributions),
                    llm_narrative=narrative,
                    is_counterfactual=False,
                )
            )

            previous_contributions = contributions

            # Apply Monte Carlo perturbation for next step
            if step < n_steps:
                current_features = self._perturb(current_features)

        logger.info(
            "Generated %d-step trajectory for patient_id=%s (risk: %.2f → %.2f)",
            n_steps, patient_id,
            trajectory[0].risk_score.value,
            trajectory[-1].risk_score.value,
        )
        return trajectory

    @staticmethod
    def _perturb(features: dict[str, float]) -> dict[str, float]:
        """Apply Gaussian noise to ROOT features, then re-derive dependents."""
        perturbed = dict(features)
        for feat, cfg in _PERTURBATION_CONFIG.items():
            if feat in perturbed:
                noise = random.gauss(0, cfg["std"])
                perturbed[feat] = max(cfg["min"], min(cfg["max"], perturbed[feat] + noise))

        # Re-derive dependent features from perturbed roots
        ap_hi = perturbed.get("ap_hi", 120)
        ap_lo = perturbed.get("ap_lo", 80)
        # Ensure ap_lo < ap_hi after independent perturbation
        if ap_lo >= ap_hi:
            ap_lo = ap_hi - 10
            perturbed["ap_lo"] = ap_lo
        perturbed["pulse_pressure"] = ap_hi - ap_lo
        perturbed["mean_arterial_pressure"] = ap_lo + (ap_hi - ap_lo) / 3.0

        height_cm = perturbed.get("height_cm", 170)
        weight_kg = perturbed.get("weight_kg", 70)
        if height_cm > 0:
            perturbed["bmi"] = round(weight_kg / ((height_cm / 100) ** 2), 4)

        return perturbed
