"""
PatientCardiovascularRecord — the core domain entity.

This dataclass is the system's canonical representation of a patient record.
It maps exactly to the confirmed dataset schema and is IMMUTABLE (frozen=True).

Design rules:
  - Zero external imports (stdlib only)
  - All fields validated at construction via __post_init__
  - Pre-engineered fields (age_years, bmi, bp_category) accepted as-is from dataset
    OR derived by the domain service when ingesting raw API payloads
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from src.domain.entities.enums import (
    BPCategory,
    CholesterolLevel,
    Gender,
    GlucoseLevel,
)


@dataclass(frozen=True)
class PatientCardiovascularRecord:
    """
    Immutable domain entity representing a single patient cardiovascular snapshot.

    Fields mirror the dataset columns:
        age       → age_days (raw integer from dataset)
        gender    → Gender enum (1=Male, 2=Female)
        height    → height_cm
        weight    → weight_kg
        ap_hi     → systolic blood pressure
        ap_lo     → diastolic blood pressure
        cholesterol → CholesterolLevel enum
        gluc      → GlucoseLevel enum
        smoke     → is_smoker
        alco      → drinks_alcohol
        active    → is_physically_active
        cardio    → has_cardiovascular_disease (ground truth label; None for live inference)

    Pre-engineered fields (from dataset or computed by domain service):
        age_years, bmi, bp_category
    """

    #Identity ──────────
    patient_id: UUID
    recorded_at: datetime

    #Raw dataset fields ─
    age_days: int
    gender: Gender
    height_cm: int
    weight_kg: float
    ap_hi: int
    ap_lo: int
    cholesterol: CholesterolLevel
    glucose: GlucoseLevel
    is_smoker: bool
    drinks_alcohol: bool
    is_physically_active: bool

    #Pre-engineered / derived fields ─────────────────────────────────────────
    age_years: float
    bmi: float
    bp_category: BPCategory

    #Ground truth label (None during live inference) ──────────────────────────
    has_cardiovascular_disease: bool | None = None

    def __post_init__(self) -> None:
        self._validate_vitals()
        self._validate_derived()

    def _validate_vitals(self) -> None:
        """Guard against clinically impossible values (dataset outlier cleaning)."""
        if not (0 < self.height_cm < 300):
            raise ValueError(f"height_cm={self.height_cm} is outside plausible range (1–299 cm)")
        if not (1 < self.weight_kg < 600):
            raise ValueError(f"weight_kg={self.weight_kg} is outside plausible range")
        if not (0 < self.ap_hi < 370):
            raise ValueError(f"ap_hi={self.ap_hi} is clinically implausible")
        if not (0 < self.ap_lo < 300):
            raise ValueError(f"ap_lo={self.ap_lo} is clinically implausible")
        if self.ap_lo >= self.ap_hi:
            raise ValueError(
                f"Diastolic ({self.ap_lo}) must be less than systolic ({self.ap_hi})"
            )
        if not (1 <= self.age_days <= 50000):
            raise ValueError(f"age_days={self.age_days} is outside plausible range")

    def _validate_derived(self) -> None:
        if not (10.0 <= self.bmi <= 80.0):
            raise ValueError(f"bmi={self.bmi} is outside plausible range (10–80)")
        if not (1.0 <= self.age_years <= 120.0):
            raise ValueError(f"age_years={self.age_years} is outside plausible range")

    #Computed properties (no mutation — derives from frozen fields) ────────────
    @property
    def pulse_pressure(self) -> int:
        """Pulse pressure = systolic − diastolic. Elevated >60 mmHg is a risk marker."""
        return self.ap_hi - self.ap_lo

    @property
    def mean_arterial_pressure(self) -> float:
        """MAP = diastolic + (pulse_pressure / 3). Clinical perfusion indicator."""
        return self.ap_lo + (self.pulse_pressure / 3.0)

    @property
    def is_hypertensive(self) -> bool:
        return self.bp_category in (
            BPCategory.HYPERTENSION_STAGE_1,
            BPCategory.HYPERTENSION_STAGE_2,
            BPCategory.HYPERTENSIVE_CRISIS,
        )

    def to_feature_dict(self) -> dict[str, float]:
        """
        Returns the ML-ready feature dictionary.
        Used by FeatureStore and ScikitLearn/LightGBM adapters.
        """
        return {
            "age_years": self.age_years,
            "gender": float(self.gender.value),
            "height_cm": float(self.height_cm),
            "weight_kg": self.weight_kg,
            "ap_hi": float(self.ap_hi),
            "ap_lo": float(self.ap_lo),
            "cholesterol": float(self.cholesterol.value),
            "glucose": float(self.glucose.value),
            "is_smoker": float(self.is_smoker),
            "drinks_alcohol": float(self.drinks_alcohol),
            "is_physically_active": float(self.is_physically_active),
            "bmi": self.bmi,
            "pulse_pressure": float(self.pulse_pressure),
            "mean_arterial_pressure": self.mean_arterial_pressure,
            "bp_category_encoded": float(
                {
                    BPCategory.NORMAL: 0,
                    BPCategory.ELEVATED: 1,
                    BPCategory.HYPERTENSION_STAGE_1: 2,
                    BPCategory.HYPERTENSION_STAGE_2: 3,
                    BPCategory.HYPERTENSIVE_CRISIS: 4,
                }[self.bp_category]
            ),
        }
