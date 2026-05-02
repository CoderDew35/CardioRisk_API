"""
Pydantic request/response schemas for all API endpoints.

Separated from routers so schemas can be imported independently
(e.g., for client SDK generation, testing, or documentation).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Patient ──────────────────────────────────────────────────────────────────

class PatientSummary(BaseModel):
    patient_id: str
    age_years: float
    gender: int
    height_cm: int
    weight_kg: float
    ap_hi: int
    ap_lo: int
    bmi: float
    bp_category: str
    cholesterol: int
    glucose: int
    is_smoker: bool
    drinks_alcohol: bool
    is_physically_active: bool
    has_cardiovascular_disease: bool | None = None


class PatientListResponse(BaseModel):
    patients: list[PatientSummary]
    total: int
    offset: int
    limit: int


# ── Ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    raw_payload: dict[str, Any] = Field(
        ...,
        example={
            "age": 18393, "gender": 2, "height": 168, "weight": 62,
            "ap_hi": 110, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1, "cardio": 0,
        }
    )
    patient_id: str | None = None
    source: str = "api"


class IngestResponse(BaseModel):
    patient_id: str
    status: str


# ── Risk ─────────────────────────────────────────────────────────────────────

class RiskResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    model_version: str
    llm_narrative: str | None = None


# ── SHAP ─────────────────────────────────────────────────────────────────────

class SHAPContributionResponse(BaseModel):
    feature: str
    value: float
    shap: float
    delta: float | None = None
    direction: str


class SHAPResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    shap_contributions: list[SHAPContributionResponse]


# ── Trajectory ───────────────────────────────────────────────────────────────

class TrajectoryPointResponse(BaseModel):
    time_step: int
    timestamp: str
    risk_score: float
    risk_percentage: float
    risk_level: str
    shap_contributions: list[SHAPContributionResponse]
    llm_narrative: str | None = None
    is_counterfactual: bool = False
    counterfactual_label: str | None = None


class TrajectoryResponse(BaseModel):
    patient_id: str
    n_steps: int
    trajectory: list[TrajectoryPointResponse]


# ── Counterfactual ───────────────────────────────────────────────────────────

class CounterfactualRequest(BaseModel):
    feature_overrides: dict[str, float] = Field(
        ...,
        example={"ap_hi": 130.0, "smoke": 0.0}
    )
    include_narrative: bool = True


class CounterfactualResponse(BaseModel):
    patient_id: str
    intervention: str
    risk_delta: float
    baseline: TrajectoryPointResponse
    counterfactual: TrajectoryPointResponse


# ── Cohort ───────────────────────────────────────────────────────────────────

class CohortAggregates(BaseModel):
    total_patients: int
    mean_age_years: float
    mean_bmi: float
    mean_ap_hi: float
    mean_ap_lo: float
    smoker_pct: float
    active_pct: float
    cvd_positive_pct: float
    bp_category_distribution: dict[str, int]
    risk_band_note: str = "Risk bands require model inference — use /patients/{id}/risk"


# ── MLOps ────────────────────────────────────────────────────────────────────

class ModelRegistryEntry(BaseModel):
    version: str
    stage: str
    auc_roc: float
    auprc: float | None = None
    brier_score: float | None = None
    created_at: str


class MLOpsStatusResponse(BaseModel):
    current_model_version: str
    model_name: str
    last_drift_check: str | None = None
    drift_detected: bool = False
    drift_scores: dict[str, float] | None = None
    is_training: bool = False

